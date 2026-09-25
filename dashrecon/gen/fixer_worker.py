"""NVIDIA Fixer worker process (runs in ``.venvs/fixer``; DECISIONS D14). Started by dashrecon.gen.fixer_client.

Fixer (nv-tlabs/Fixer @ b39dfca, weights ``nvidia/Fixer``) is a single-step restorer built on Cosmos-Predict2 0.6B:
the image is VAE-encoded without added noise, denoised once by the DiT at sigma = timestep / 1000 and decoded
(``Pix2Pix_Turbo.forward``). Inputs are RGB in [-1, 1] (``inference_pretrained_model.preprocess_image``), bf16.
Height and width must be multiples of 16; the dashrecon renders (960x640) are used at their own size (val056 /
val041 test: same result as the 1024x576 default without changing the aspect ratio).

Protocol, one JSON object per line. The worker prints ``{"ready": ...}`` once loaded, then for every request
``{"in": "<file>.npy", "out": "<file>.npy"}`` ((N, H, W, 3) float16 in [0, 1]) writes the refined images with the
same shape and prints ``{"ok": N, "seconds": ...}``. Everything the libraries print goes to stderr, so stdout only
carries the protocol.
"""
import argparse
import json
import os
import sys
import time

# keep a private copy of stdout for the protocol and send every other print (Python or C level) to stderr
PROTOCOL = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import torch  # noqa: E402


def say(obj: dict) -> None:
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixer_src", required=True, help="checkout of nv-tlabs/Fixer")
    parser.add_argument("--weights_dir", required=True, help="snapshot of nvidia/Fixer (base/, pretrained/)")
    parser.add_argument("--timestep", type=int, required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.fixer_src, "src"))
    import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as fx

    fx.config.dit_path = os.path.join(args.weights_dir, "base", "model_fast_tokenizer.pt")
    fx.config.tokenizer["vae_pth"] = os.path.join(args.weights_dir, "base", "tokenizer_fast.pth")
    torch.set_grad_enabled(False)
    t0 = time.time()
    model = fx.Pix2Pix_Turbo(pretrained_path=os.path.join(args.weights_dir, "pretrained", "pretrained_fixer.pkl"),
                             timestep=args.timestep, vae_skip_connection=False, batch_size=1)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.set_eval()
    say({"ready": True, "load_seconds": time.time() - t0, "timestep": args.timestep, "torch": torch.__version__})

    for line in sys.stdin:
        req = json.loads(line)
        t1 = time.time()
        imgs = np.load(req["in"])
        assert imgs.ndim == 4 and imgs.shape[-1] == 3 and imgs.shape[1] % 16 == 0 and imgs.shape[2] % 16 == 0, imgs.shape
        out = np.empty_like(imgs, dtype=np.float16)
        for i, im in enumerate(imgs):
            x = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1)[None].cuda() * 2.0 - 1.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model(x.to(torch.bfloat16))
            out[i] = ((y.float()[0].permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1).cpu().numpy().astype(np.float16)
        np.save(req["out"], out)
        say({"ok": int(len(imgs)), "seconds": time.time() - t1})


if __name__ == "__main__":
    main()
