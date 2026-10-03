"""GPU post-processing for the Grounded-SAM-2 dynamic masks of E34 (docs/E34_DATA.md; ``.venvs/masks``).

Same models, prompts, thresholds and dilation as ``dashrecon.masks.sam_text.GroundedSam2Backend``; only where the work
runs changes, because the E34 data needs about 600 images per segment:
    - SAM's low-resolution masks are upsampled on the GPU (``Sam2Processor.post_process_masks`` on CUDA tensors, the
      same bilinear interpolation and > 0 threshold) instead of on the CPU;
    - the disk dilation is a convolution with the same disk on the GPU (> 0.5), equal to scipy's binary_dilation with
      border value 0 (the disk is symmetric, sums of 0/1 values are exact).
Checked against the original backend on 30 images of wt683 (FRONT, FRONT_LEFT, SIDE_LEFT): identical masks in fp32.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dashrecon.masks.sam_text import GroundedSam2Backend, disk


def dilate_gpu(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a (H,W) bool CUDA mask by a disk of ``radius`` pixels (radius 0 returns it unchanged)."""
    assert radius >= 0, radius
    if radius == 0:
        return mask
    k = torch.from_numpy(disk(radius).astype(np.float32)).to(mask.device)[None, None]
    return F.conv2d(mask.float()[None, None], k, padding=radius)[0, 0] > 0.5


class GroundedSam2Gpu(GroundedSam2Backend):
    """GroundedSam2Backend with the mask upsampling and dilation on the GPU."""

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[dict[str, np.ndarray], dict]:
        w, h = image.size
        det_in = self.det_proc(images=image, text=self.text, return_tensors="pt").to(self.device)
        det_out = self.det(**det_in)
        dets = self.det_proc.post_process_grounded_object_detection(
            det_out, det_in.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = dets["boxes"].cpu().tolist()
        labels = list(dets["text_labels"])
        mask = torch.zeros((h, w), dtype=torch.bool, device=self.device)
        if boxes:
            sam_in = self.sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            sam_out = self.sam(**sam_in, multimask_output=False)
            masks = self.sam_proc.post_process_masks(sam_out.pred_masks, sam_in["original_sizes"])[0]
            assert tuple(masks.shape) == (len(boxes), 1, h, w), tuple(masks.shape)
            mask = masks[:, 0].any(dim=0)
        counts = {p: 0 for p in self.prompts}
        for lab in labels:
            for p in self.prompts:
                if p in lab.split():
                    counts[p] += 1
        return {"dynamic": dilate_gpu(mask, self.dilate_px).cpu().numpy()}, {"num_boxes": len(boxes), "counts": counts}
