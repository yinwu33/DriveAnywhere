"""Exercise the actual upstream real-frame sampler with our scene seed policy."""
from types import SimpleNamespace

import numpy as np
import torch

from datasets.base.pixel_source import ScenePixelSource
from dashrecon.train.seeds import seed_scene_optimization


def sample_sequence(seed: int) -> tuple[list[int], list[int], list[int]]:
    """Sample real-frame indices and the NumPy/Torch streams without scene data."""
    seed_scene_optimization(seed)
    sampler = SimpleNamespace(buffer_ratio=0.0, image_error_buffered=False)
    candidates = torch.arange(20)
    frames = [int(ScenePixelSource.propose_training_image(sampler, candidates)) for _ in range(40)]
    return frames, np.random.randint(0, 100, 40).tolist(), torch.randint(0, 100, (40,)).tolist()


def test_real_frame_sampler_and_other_streams_repeat() -> None:
    assert sample_sequence(0) == sample_sequence(0)
    assert sample_sequence(0) != sample_sequence(1)
