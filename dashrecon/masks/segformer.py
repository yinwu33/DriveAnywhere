"""Sky and road masks from SegFormer-B5 Cityscapes (DECISIONS D6).

Hugging Face ``nvidia/segformer-b5-finetuned-cityscapes-1024-1024``; class ids are checked against
the model config at load time (Cityscapes: 0 = road, 10 = sky). The processor's default 1024x1024
resize would squash 3:2 frames, so images are resized to ``input_hw`` (aspect-preserving by choice)
and logits are upsampled back with ``post_process_semantic_segmentation(target_sizes=...)``.
"""
import numpy as np
import torch
from PIL import Image

from dashrecon.masks.base import MaskBackend

CLASS_IDS = {"road": 0, "sky": 10}


class SegformerSkyRoadBackend(MaskBackend):
    """Sky and road masks from one Cityscapes semantic segmentation pass."""

    name = "segformer"
    kinds = ("sky", "road")

    def __init__(self, model_id: str, input_hw: tuple[int, int]) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = torch.device("cuda")
        self.model_id, self.input_hw = model_id, input_hw
        self.proc = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device).eval()
        for kind, cid in CLASS_IDS.items():
            assert self.model.config.id2label[cid] == kind, (cid, self.model.config.id2label[cid])

    @torch.no_grad()
    def labels(self, image: Image.Image) -> np.ndarray:
        """Cityscapes class id of every pixel (H, W), at the image's own size."""
        w, h = image.size
        inputs = self.proc(images=image, size={"height": self.input_hw[0], "width": self.input_hw[1]},
                           return_tensors="pt").to(self.device)
        return self.proc.post_process_semantic_segmentation(self.model(**inputs), target_sizes=[(h, w)])[0].cpu().numpy()

    def class_id(self, name: str) -> int:
        """Id of a Cityscapes class name in the loaded model's config (KeyError if the model has no such class)."""
        return int(self.model.config.label2id[name])

    def predict(self, image: Image.Image) -> tuple[dict[str, np.ndarray], dict]:
        seg = self.labels(image)
        return {kind: seg == cid for kind, cid in CLASS_IDS.items()}, {}

    def meta(self) -> dict:
        import transformers

        return {
            "backend": self.name,
            "model_id": self.model_id,
            "input_hw": list(self.input_hw),
            "class_ids": CLASS_IDS,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        }
