"""Client for the NVIDIA Fixer worker (DECISIONS D14): runs dashrecon/gen/fixer_worker.py in ``.venvs/fixer``.

Fixer needs torch 2.6 + Cosmos-Predict2 while drivestudio training runs on torch 2.4.1 + gsplat 1.3.0, so the two
live in separate venvs (AGENTS.md section 9) and exchange images through .npy files in ``work_dir``.
"""
import glob
import json
import os
import subprocess
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXER_VENV = os.path.join(REPO, ".venvs", "fixer")
FIXER_SRC = os.path.join(REPO, ".venvs", "src", "Fixer")
FIXER_REPO_ID = "nvidia/Fixer"
FIXER_REVISION = "ca20a25b55baa15dfc557db4c9b99168da8c2e14"


class FixerClient:
    """A running Fixer worker; ``refine`` sends a batch of images and waits for the result."""

    def __init__(self, timestep: int, work_dir: str, cuda_home: str) -> None:
        from huggingface_hub import snapshot_download

        weights = snapshot_download(FIXER_REPO_ID, revision=FIXER_REVISION, local_files_only=True)
        libs = sorted(glob.glob(os.path.join(FIXER_VENV, "lib", "python3.12", "site-packages", "nvidia", "*", "lib")))
        assert libs, f"no nvidia libraries in {FIXER_VENV}: run envs/setup_fixer.sh"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(libs)  # transformer_engine loads cuDNN through ctypes
        env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + env["PATH"]
        env["HF_HUB_OFFLINE"] = "1"
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.log = open(os.path.join(work_dir, "fixer_worker.log"), "w")
        self.proc = subprocess.Popen(
            [os.path.join(FIXER_VENV, "bin", "python"), os.path.join(REPO, "dashrecon", "gen", "fixer_worker.py"),
             "--fixer_src", FIXER_SRC, "--weights_dir", weights, "--timestep", str(timestep)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)
        self.info = self._read()
        assert self.info["ready"], self.info
        self.timestep, self.calls, self.seconds = timestep, 0, 0.0

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, f"Fixer worker exited (code {self.proc.poll()}); see {self.log.name}"
        return json.loads(line)

    def refine(self, images: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) float in [0, 1] -> refined (N, H, W, 3) float32 in [0, 1]."""
        t0 = time.time()
        src, dst = os.path.join(self.work_dir, "in.npy"), os.path.join(self.work_dir, "out.npy")
        np.save(src, images.astype(np.float16))
        self.proc.stdin.write(json.dumps({"in": src, "out": dst}) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        assert reply["ok"] == len(images), reply
        out = np.load(dst).astype(np.float32)
        self.calls += 1
        self.seconds += time.time() - t0
        return out

    def close(self) -> None:
        self.proc.stdin.close()
        code = self.proc.wait()
        self.log.close()
        assert code == 0, f"Fixer worker exited with {code}; see {self.log.name}"
