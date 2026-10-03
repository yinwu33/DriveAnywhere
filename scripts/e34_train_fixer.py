"""E34 (docs/PLAN_AFTER_E30.md, docs/E34_DATA.md): fine-tune NVIDIA Fixer on the driving pairs. Runs in .venvs/fixer_train
(envs/setup_fixer_train.sh: the Fixer env plus the official training dependencies and tokenizer patch).

The official recipe (Fixer README "Training", src/train_pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py) on a single
GPU, with two changes:
    loss mask   every pair carries a mask (255 = pixel left out: the target camera's dynamic objects, undistortion
                border). L2 is averaged over the kept pixels; for LPIPS the left-out target pixels are replaced by the
                detached prediction, so they give no gradient (scripts/train_fixer.py does the same with its keep mask).
                The official loader has no masks, and E24 showed that cars in the targets teach the model to paint cars.
    no trackers wandb, the discriminator (lambda_gan 0 in the official command), CLIP and Gram terms are left out;
                losses go to <output_dir>/log.jsonl, checkpoints to <output_dir>/checkpoints/model_<step>.pkl with the
                keys Pix2Pix_Turbo(pretrained_path=...) and dashrecon/gen/fixer_worker.py --pretrained_pkl read
                (state_dict_unet, state_dict_vae) plus the optimizer for --resume.
Everything else as the official command: full DiT + VAE (time_conv frozen) in bf16 weights, timestep 250, AdamW 2e-5
(betas 0.9 / 0.999, weight decay 1e-2), constant schedule with 500 warm-up steps, grad norm 1, L2 weight 1 and LPIPS-VGG
weight 0.3 on a random 128-512 px crop, images resized to 576 x 1024 (Lanczos, aspect not kept) and scaled to [-1, 1],
batch 1. Weights start from nvidia/Fixer's pretrained_fixer.pkl (--weights_dir, as the worker).

Pairs: --dataset is the Fixer-format json of scripts/e34_manifest.py ({"train" | "test": {id: {image, target_image, prompt,
mask}}}); several files may be given (comma separated) and are concatenated. Evaluation every --eval_every steps on the
first --eval_samples test pairs: PSNR over kept pixels and LPIPS-Alex (the metric of the rest of the project) of
prediction vs target, and <output_dir>/eval/<step>/<k>.jpg (input | prediction | target) for the first --eval_images.

Example:
    CUDA_HOME=$HOME/cuda-12.6 .venvs/fixer_train/bin/python scripts/e34_train_fixer.py --dataset data/e34/fixer_gs.json \\
        --weights_dir ~/.cache/huggingface/hub/models--nvidia--Fixer/snapshots/<rev> --output_dir results/E34/fixer_gs \\
        --max_steps 10000 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dashrecon.provenance import git_commit  # noqa: E402

FIXER_SRC = os.path.join(REPO, ".venvs", "src", "Fixer", "src")
HW = (576, 1024)


class MaskedPairs(torch.utils.data.Dataset):
    """Fixer pairs with a loss mask (255 = left out), resized as the official "resize_576x1024" prep."""

    def __init__(self, json_paths: list, split: str) -> None:
        self.items = []
        for p in json_paths:
            with open(p) as f:
                data = json.load(f)[split]
            self.items += [data[k] for k in sorted(data)]
        assert self.items, f"no {split} pairs in {json_paths}"
        self.resize = transforms.Resize(HW, interpolation=transforms.InterpolationMode.LANCZOS)

    def __len__(self) -> int:
        return len(self.items)

    def image(self, path: str) -> torch.Tensor:
        return transforms.functional.normalize(transforms.functional.to_tensor(self.resize(Image.open(path).convert("RGB"))), [0.5], [0.5])

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        mask = np.asarray(Image.open(it["mask"]).convert("L").resize((HW[1], HW[0]), Image.NEAREST)) > 127
        return {"src": self.image(it["image"]), "tgt": self.image(it["target_image"]),
                "keep": torch.from_numpy(~mask)[None].float(), "id": os.path.basename(it["target_image"])}


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Fixer-format json file(s), comma separated")
    parser.add_argument("--weights_dir", required=True, help="snapshot of nvidia/Fixer (base/, pretrained/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=40)
    parser.add_argument("--eval_images", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", help="checkpoints/model_<step>.pkl of this script")
    args = parser.parse_args()
    commit = git_commit()
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    sys.path.insert(0, FIXER_SRC)
    import lpips
    import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as fx

    fx.config.dit_path = os.path.join(args.weights_dir, "base", "model_fast_tokenizer.pt")
    fx.config.tokenizer["vae_pth"] = os.path.join(args.weights_dir, "base", "tokenizer_fast.pth")
    pretrained = os.path.join(args.weights_dir, "pretrained", "pretrained_fixer.pkl")
    net = fx.Pix2Pix_Turbo(pretrained_path=pretrained, timestep=args.timestep, vae_skip_connection=False, batch_size=1,
                           train_full_unet=True, use_sched=True)
    net.set_train()
    params = [p for p in list(net.unet.parameters()) + list(net.vae.parameters()) if p.requires_grad]
    net.to(device, dtype=torch.bfloat16)  # the official script trains bf16 weights (accelerate mixed_precision bf16 + .to(dtype))
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    step = 0
    if args.resume is not None:
        sd = torch.load(args.resume, map_location="cpu")
        net.unet.load_state_dict(sd["state_dict_unet"])
        net.vae.load_state_dict(sd["state_dict_vae"])
        opt.load_state_dict(sd["optimizer"])
        step = int(sd["step"])
        print(f"[e34_train_fixer] resumed {args.resume} at step {step}", flush=True)

    json_paths = [p.strip() for p in args.dataset.split(",")]
    train_set, test_set = MaskedPairs(json_paths, "train"), MaskedPairs(json_paths, "test")
    loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers, drop_last=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    print(f"[e34_train_fixer] {len(train_set)} train / {len(test_set)} test pairs, {sum(p.numel() for p in params) / 1e6:.0f}M "
          f"trainable parameters, steps {step + 1}..{args.max_steps}", flush=True)
    with open(os.path.join(args.output_dir, "train_params.json"), "w") as f:
        json.dump({**vars(args), "pretrained": pretrained, "train_pairs": len(train_set), "test_pairs": len(test_set),
                   "dashrecon_commit": commit}, f, indent=2)

    @torch.no_grad()
    def evaluate(s: int) -> dict:
        net.set_eval()
        out_dir = os.path.join(args.output_dir, "eval", f"{s:06d}")
        os.makedirs(out_dir, exist_ok=True)
        psnr, lp = [], []
        for k in range(min(args.eval_samples, len(test_set))):
            b = test_set[k]
            src, tgt, keep = b["src"][None].to(device, torch.bfloat16), b["tgt"][None].to(device), b["keep"][None].to(device)
            pred = net(src).float()
            mse = ((pred - tgt) ** 2 * keep).sum() / (3.0 * keep.sum()).clamp(min=1.0)
            psnr.append(float(-10 * torch.log10(mse / 4.0)))  # [-1, 1] range -> 2 ** 2
            lp.append(float(lpips_alex(pred, torch.where(keep > 0.5, tgt, pred))))
            if k < args.eval_images:
                Image.fromarray(np.concatenate([to_uint8(src[0]), to_uint8(pred[0]), to_uint8(tgt[0])], 1)).save(
                    os.path.join(out_dir, f"{k:02d}_{b['id']}"), quality=88)
        net.set_train()
        return {"step": s, "test_psnr_kept": float(np.mean(psnr)), "test_lpips_alex": float(np.mean(lp)), "samples": len(psnr)}

    def save(s: int) -> None:
        torch.save({"state_dict_unet": net.unet.state_dict(), "state_dict_vae": net.vae.state_dict(), "optimizer": opt.state_dict(),
                    "step": s, "timestep": args.timestep, "dashrecon_commit": commit},
                   os.path.join(args.output_dir, "checkpoints", f"model_{s}.pkl"))

    log = open(os.path.join(args.output_dir, "log.jsonl"), "a")
    if step == 0:
        ev = evaluate(0)
        log.write(json.dumps(ev) + "\n")
        print(f"[e34_train_fixer] before training: PSNR {ev['test_psnr_kept']:.2f}  LPIPS {ev['test_lpips_alex']:.3f}", flush=True)
    running, t0 = {}, time.time()
    while step < args.max_steps:
        for b in loader:
            if step >= args.max_steps:
                break
            step += 1
            src, tgt, keep = b["src"].to(device, torch.bfloat16), b["tgt"].to(device, torch.bfloat16), b["keep"].to(device, torch.bfloat16)
            pred = net(src)
            l2 = ((pred - tgt) ** 2 * keep).sum() / (3.0 * keep.sum()).clamp(min=1.0)
            tgt_eff = torch.where(keep > 0.5, tgt, pred.detach())
            h, w = pred.shape[-2:]
            ch = cw = random.randint(128, 512)
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            lp = lpips_vgg(crop(pred, top, left, ch, cw), crop(tgt_eff, top, left, ch, cw)).mean()
            loss = args.lambda_l2 * l2 + args.lambda_lpips * lp
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at step {step}: l2 {float(l2)} lpips {float(lp)}")
            lr = args.learning_rate * min(1.0, step / args.lr_warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            for k, v in (("loss", loss), ("l2", l2), ("lpips", lp), ("keep", keep.mean())):
                running[k] = running.get(k, 0.0) + float(v)
            if step % args.print_every == 0:
                msg = {k: v / args.print_every for k, v in running.items()}
                log.write(json.dumps({"step": step, **msg, "lr": lr}) + "\n")
                log.flush()
                print(f"[e34_train_fixer] step {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in msg.items())
                      + f"  {args.print_every / (time.time() - t0):.2f} it/s  {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                running, t0 = {}, time.time()
            if step % args.eval_every == 0 or step == args.max_steps:
                ev = evaluate(step)
                log.write(json.dumps(ev) + "\n")
                log.flush()
                print(f"[e34_train_fixer] eval {step}: PSNR {ev['test_psnr_kept']:.2f}  LPIPS {ev['test_lpips_alex']:.3f}", flush=True)
            if step % args.checkpoint_every == 0 or step == args.max_steps:
                save(step)
    log.close()
    print(f"[e34_train_fixer] done: {args.output_dir}/checkpoints/model_{step}.pkl", flush=True)


if __name__ == "__main__":
    main()
