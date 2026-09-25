"""Frozen SDXL + depth ControlNet refiner for generative distillation (AGENTS Phase 8, DECISIONS D10-D12).

This is the generation step of GGDS (LSD-3D), using off-the-shelf weights with no fine-tuning:
    1. encode the rendered view with the SDXL VAE (fp16-fix), resized to ``gen_hw``;
    2. deterministic DDIM inversion to timestep T = round(t * 999) in ``num_steps`` steps, conditional branch only;
    3. ``num_steps`` DDIM steps from T back to a clean latent, with classifier-free guidance;
    4. decode, then resize back to the render resolution.
Both passes condition the UNet through the SDXL depth ControlNet on a disparity image of the NKSR mesh
(``disparity_image``), so the refined image keeps the mesh geometry. Everything runs without gradient and is
deterministic (no noise is sampled). The caller uses the result as a fixed target for the 3DGS render (E6) or
as the output image (E5+pp).

Weights (Hugging Face): stabilityai/stable-diffusion-xl-base-1.0 (CreativeML Open RAIL++-M),
diffusers/controlnet-depth-sdxl-1.0 (OpenRAIL++), madebyollin/sdxl-vae-fp16-fix (MIT).
"""
from typing import List, Tuple

import torch
import torch.nn.functional as F

BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_ID = "diffusers/controlnet-depth-sdxl-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
NUM_TRAIN_TIMESTEPS = 1000


def disparity_image(depth: torch.Tensor, valid: torch.Tensor, max_pct: float) -> torch.Tensor:
    """ControlNet input from a z-depth map [H, W]: 1/z scaled to [0, 1] (near = bright), 0 where ``valid`` is False.

    Disparity is divided by its ``max_pct`` percentile over valid pixels and clipped, so the road right in front of
    the camera saturates instead of pushing everything else into the dark end; zero disparity (sky, pixels the
    mesh does not cover) stays at 0. This mimics the relative inverse depth (DPT) the depth ControlNet was
    trained on, where the sky is black and distant structure is dark grey rather than black.
    """
    assert bool(valid.any()), "the mesh covers no pixel of this view: no ControlNet condition"
    disp = torch.zeros_like(depth)
    disp[valid] = 1.0 / depth[valid].clamp(min=1e-6)
    hi = torch.quantile(disp[valid], max_pct / 100.0)
    return (disp / hi.clamp(min=1e-6)).clamp(0.0, 1.0) * valid


class SDXLRefiner:
    """Refines a rendered image towards the SDXL image prior, with the mesh disparity as ControlNet condition."""

    def __init__(self, device: torch.device, prompt: str, negative_prompt: str, guidance_scale: float,
                 controlnet_scale: float, num_steps: int, gen_hw: Tuple[int, int]) -> None:
        from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

        h, w = gen_hw
        assert h % 8 == 0 and w % 8 == 0, gen_hw
        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype, variant="fp16")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_ID, vae=vae, controlnet=controlnet, torch_dtype=dtype, variant="fp16").to(device)
        with torch.no_grad():
            pe, npe, ppe, nppe = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt, device=device,
                                                    num_images_per_prompt=1, do_classifier_free_guidance=True)
        # batch order for classifier-free guidance: [unconditional, conditional]
        self.embeds = torch.cat([npe, pe])
        self.pooled = torch.cat([nppe, ppe])
        # SDXL micro-conditioning: original size, crop top-left, target size
        self.time_ids = torch.tensor([[h, w, 0, 0, h, w]] * 2, dtype=dtype, device=device)
        self.unet, self.controlnet, self.vae = pipe.unet, pipe.controlnet, pipe.vae
        for m in (self.unet, self.controlnet, self.vae):
            m.eval().requires_grad_(False)
        del pipe  # drops both text encoders
        torch.cuda.empty_cache()
        # scaled_linear schedule of the SDXL base scheduler config
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, NUM_TRAIN_TIMESTEPS, dtype=torch.float64) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).tolist()
        self.device, self.dtype = device, dtype
        self.guidance_scale, self.controlnet_scale, self.num_steps, self.gen_hw = guidance_scale, controlnet_scale, num_steps, gen_hw

    def timesteps(self, t: float) -> List[int]:
        """Descending DDIM timesteps from round(t * 999) in ``num_steps`` equal steps (the last one above 0)."""
        assert 0.0 < t <= 1.0, t
        top = round(t * (NUM_TRAIN_TIMESTEPS - 1))
        return [round(top * (self.num_steps - i) / self.num_steps) for i in range(self.num_steps)]

    def signal_weight(self, t: float) -> float:
        """sqrt(alpha_bar) at the start timestep: the share of the render that survives the noising."""
        return self.alphas_cumprod[self.timesteps(t)[0]] ** 0.5

    def _eps(self, z: torch.Tensor, t: int, control: torch.Tensor, guided: bool) -> torch.Tensor:
        if guided:
            z, control = torch.cat([z, z]), torch.cat([control, control])
            emb, pooled, time_ids = self.embeds, self.pooled, self.time_ids
        else:
            emb, pooled, time_ids = self.embeds[1:], self.pooled[1:], self.time_ids[1:]
        added = {"text_embeds": pooled, "time_ids": time_ids}
        ts = torch.tensor(t, device=self.device)
        down, mid = self.controlnet(z, ts, encoder_hidden_states=emb, controlnet_cond=control,
                                    conditioning_scale=self.controlnet_scale, added_cond_kwargs=added, return_dict=False)
        eps = self.unet(z, ts, encoder_hidden_states=emb, added_cond_kwargs=added, down_block_additional_residuals=down,
                        mid_block_additional_residual=mid, return_dict=False)[0].float()
        if not guided:
            return eps
        uncond, cond = eps.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)

    @torch.no_grad()
    def refine(self, rgb: torch.Tensor, control: torch.Tensor, t: float) -> torch.Tensor:
        """rgb [H, W, 3] in [0, 1], control [H, W] in [0, 1] (``disparity_image``) -> refined rgb [H, W, 3] in [0, 1]."""
        H, W = rgb.shape[:2]
        h, w = self.gen_hw
        x = F.interpolate(rgb.permute(2, 0, 1)[None].float(), (h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        c = F.interpolate(control[None, None].float(), (h, w), mode="bilinear", align_corners=False)
        c = c.expand(-1, 3, -1, -1).to(self.dtype)
        sf = self.vae.config.scaling_factor
        z = self.vae.encode((2 * x - 1).to(self.dtype)).latent_dist.mean.float() * sf
        ts, a = self.timesteps(t), self.alphas_cumprod
        a_cur = 1.0
        for tt in reversed(ts):  # DDIM inversion: clean -> ts[-1] -> ... -> ts[0]
            eps = self._eps(z.to(self.dtype), tt, c, guided=False)
            x0 = (z - (1 - a_cur) ** 0.5 * eps) / a_cur ** 0.5
            a_cur = a[tt]
            z = a_cur ** 0.5 * x0 + (1 - a_cur) ** 0.5 * eps
        for i, tt in enumerate(ts):  # DDIM sampling: ts[0] -> ... -> ts[-1] -> clean
            eps = self._eps(z.to(self.dtype), tt, c, guided=True)
            x0 = (z - (1 - a[tt]) ** 0.5 * eps) / a[tt] ** 0.5
            a_prev = a[ts[i + 1]] if i + 1 < len(ts) else 1.0
            z = a_prev ** 0.5 * x0 + (1 - a_prev) ** 0.5 * eps
        img = self.vae.decode((z / sf).to(self.dtype)).sample.float()
        img = ((img + 1) / 2).clamp(0, 1)
        img = F.interpolate(img, (H, W), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return img[0].permute(1, 2, 0)
