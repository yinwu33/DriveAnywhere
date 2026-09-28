"""Disk-separated experiment stages (as scripts/run_e14.py): every stage is one CLI in a project venv, logged to
<out>/logs/<stage>.log; <out>/run_status.json records each command, exit code, time and commit; the run stops at
the first failure. With resume=True a stopped run continues: the config must be unchanged, stages that succeeded
are skipped when their command is identical, everything from the failed stage on runs again with the current
commit (recorded per stage and in run_status.json "resumes")."""
import json
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf


class StageRunner:
    def __init__(self, out: Path, cfg: dict, commit: str, resume: bool, tag: str) -> None:
        self.out, self.commit, self.tag = out, commit, tag
        self.done = {}
        if resume:
            assert OmegaConf.to_container(OmegaConf.load(out / "config.yaml"), resolve=True) == cfg, "config changed"
            self.state = json.loads((out / "run_status.json").read_text())
            assert not self.state["completed"], f"{out} is complete"
            # a stage entry has no exit_code when the runner itself died while the stage ran
            self.done = {e["name"]: e for e in self.state["stages"] if "exit_code" in e and e["exit_code"] == 0}
            self.state["stages"] = list(self.done.values())
            if "resumes" not in self.state:
                self.state["resumes"] = []
            self.state["resumes"].append({"time": time.time(), "dashrecon_commit": commit, "skipped": sorted(self.done)})
        else:
            out.mkdir(parents=True, exist_ok=False)
            (out / "logs").mkdir()
            OmegaConf.save(OmegaConf.create(cfg), out / "config.yaml")
            self.state = {"completed": False, "started": time.time(), "stages": [], "dashrecon_commit": commit}
        self._save()

    def _save(self) -> None:
        (self.out / "run_status.json").write_text(json.dumps(self.state, indent=2))

    def run(self, name: str, environment: str, arguments: list[str]) -> None:
        """Run one CLI stage in .venvs/<environment>; raise if it fails."""
        command = [f".venvs/{environment}/bin/python", "-u", *arguments]
        if name in self.done:
            assert self.done[name]["command"] == command, f"{name}: command differs from the successful run"
            print(f"[{self.tag}] SKIP {name} (succeeded before)", flush=True)
            return
        entry = {"name": name, "command": command, "started": time.time(), "log": str(self.out / "logs" / f"{name}.log"),
                 "dashrecon_commit": self.commit}
        self.state["stages"].append(entry)
        self._save()
        print(f"[{self.tag}] START {name} -> {entry['log']} ({time.strftime('%H:%M')})", flush=True)
        with open(entry["log"], "w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        entry.update(exit_code=process.returncode, elapsed_s=time.time() - entry["started"])
        self._save()
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit {process.returncode}: {entry['log']}")
        print(f"[{self.tag}] DONE {name} {entry['elapsed_s'] / 60:.1f} min", flush=True)

    def complete(self) -> None:
        self.state.update(completed=True, runtime_s=time.time() - self.state["started"])
        self._save()
