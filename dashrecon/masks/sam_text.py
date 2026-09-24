"""Text-prompted dynamic-object masks: Grounding DINO boxes -> SAM 2.1 masks (DECISIONS D5).

Implemented with the Hugging Face ``transformers`` ports of both models (same checkpoints as the
IDEA-Research Grounded-SAM-2 repo). APIs confirmed in transformers 5.17.0:
    - ``GroundingDinoProcessor.post_process_grounded_object_detection(outputs, input_ids, threshold,
      text_threshold, target_sizes)`` -> boxes in x0,y0,x1,y1 pixels of the original image;
    - ``Sam2Processor(images, input_boxes=[boxes])`` + ``Sam2Model(..., multimask_output=False)`` +
      ``Sam2Processor.post_process_masks(pred_masks, original_sizes)`` -> (num_boxes, 1, H, W) bool.
Detection runs at the processor's default resolution (shortest edge 800): at the full 1280x1920
input Grounding DINO returned no boxes at all on the dev scenes. Frames are processed independently
(no video tracking).
"""
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from dashrecon.masks.base import MaskBackend

DEFAULT_PROMPTS = ("car", "truck", "bus", "motorcycle", "bicycle", "person")


def disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given pixel radius."""
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask by ``radius`` pixels (radius 0 returns the mask unchanged)."""
    assert radius >= 0, radius
    if radius == 0:
        return mask
    return binary_dilation(mask, structure=disk(radius))


class GroundedSam2Backend(MaskBackend):
    """Union of SAM 2.1 masks for every Grounding DINO box matching the text prompts."""

    name = "gsam2"
    kinds = ("dynamic",)

    def __init__(self, detector_id: str, sam_id: str, prompts: tuple[str, ...], box_threshold: float,
                 text_threshold: float, dilate_px: int) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.device = torch.device("cuda")
        self.detector_id, self.sam_id = detector_id, sam_id
        self.prompts = prompts
        self.text = ". ".join(prompts) + "."  # Grounding DINO expects lower-case, period-separated phrases
        self.box_threshold, self.text_threshold, self.dilate_px = box_threshold, text_threshold, dilate_px
        self.det_proc = AutoProcessor.from_pretrained(detector_id)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id).to(self.device).eval()

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
        mask = np.zeros((h, w), dtype=bool)
        if boxes:
            sam_in = self.sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
            sam_out = self.sam(**sam_in, multimask_output=False)
            masks = self.sam_proc.post_process_masks(sam_out.pred_masks.cpu(), sam_in["original_sizes"].cpu())[0]
            assert tuple(masks.shape) == (len(boxes), 1, h, w), tuple(masks.shape)
            mask = masks[:, 0].any(dim=0).numpy()
        counts = {p: 0 for p in self.prompts}
        for lab in labels:
            for p in self.prompts:
                if p in lab.split():
                    counts[p] += 1
        return {"dynamic": dilate(mask, self.dilate_px)}, {"num_boxes": len(boxes), "counts": counts}

    def meta(self) -> dict:
        import transformers

        return {
            "backend": self.name,
            "detector_id": self.detector_id,
            "sam_id": self.sam_id,
            "prompts": list(self.prompts),
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "dilate_px": self.dilate_px,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }
