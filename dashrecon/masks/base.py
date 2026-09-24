"""Abstract interface of mask backends (AGENTS.md Phase 4)."""
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class MaskBackend(ABC):
    """Maps one RGB image to boolean masks of fixed kinds (see ``dashrecon.io.MASK_KINDS``)."""

    name: str
    kinds: tuple[str, ...]

    @abstractmethod
    def predict(self, image: Image.Image) -> tuple[dict[str, np.ndarray], dict]:
        """Predict masks for one image.

        Args:
            image: RGB image at original resolution.

        Returns:
            (masks, info): ``masks[kind]`` is an (H,W) bool array at the image resolution for every kind
            in ``self.kinds``; ``info`` holds per-frame diagnostics (e.g. detection counts).
        """

    @abstractmethod
    def meta(self) -> dict:
        """Model ids, versions and parameters for meta.json."""
