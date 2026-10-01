"""Per-image exposure as MTGS has it (MT1e / MT1x / MT1m in docs/EXPERIMENTS.md).

drivestudio's AffineTransform (models/modules.py) decodes a per-image embedding with a small MLP, and zero-initialises
the embedding and every layer. The gradients of the embedding, of the first layer and of the second layer's weights
are then exactly zero (the hidden units stay at ReLU(0) = 0 and the second layer's weights are 0), so only the final
bias learns: one colour offset shared by every image. Every run so far had no per-image exposure at all (checked on
the MT1 / MT1A checkpoints: embedding and weights all 0, only decoder.2.bias non-zero). Two days of light need it.

PerImageExposure is MTGS's LearnableExposureRGBModel (mtgs/scene_model/module/appearance.py): one 3 x 4 colour matrix
per image, initialised to [I | 0] and learned directly, applied by drivestudio's affine_transformation like the
original module. A held-out image never gets gradients, so in the test set (``in_test_set``, set by drivestudio while
rendering the test split) an image uses the matrix of the image before it: with test_image_stride > 1 counted from
the first frame (DECISIONS D7) that is a training frame of the same traversal, a tenth of a second earlier.
"""
import torch
from torch.nn import Parameter


class PerImageExposure(torch.nn.Module):
    """A learned 3 x 4 colour transform per image (drop-in for models.modules.AffineTransform)."""

    def __init__(self, class_name: str, n: int, device: torch.device = torch.device("cuda"), **kwargs) -> None:
        super().__init__()
        self.class_prefix = class_name + "#"
        self.exposure = Parameter(torch.eye(3, 4, device=device)[None].repeat(n, 1, 1))
        self.in_test_set = False

    def forward(self, image_infos) -> torch.Tensor:
        idx = image_infos["img_idx"]
        if self.in_test_set:
            assert int(idx.min()) >= 1, "a held-out first image has no earlier training image"
            idx = idx - 1
        return self.exposure[idx]

    def get_param_groups(self):
        return {self.class_prefix + "all": [self.exposure]}
