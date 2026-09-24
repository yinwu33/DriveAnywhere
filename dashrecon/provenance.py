"""Code provenance for meta.json / vis_params.json (AGENTS.md section 8: git commit hash)."""
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIRS = ["dashrecon", "scripts", "envs", "configs", "datasets", "models", "tools", "utils"]


def git_commit() -> str:
    """HEAD commit of this repo, suffixed with ``-dirty`` if code directories differ from it.

    Untracked files under the code directories also count as dirty, since results produced by
    uncommitted code cannot be reproduced from the hash alone.
    """
    head = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", REPO_ROOT, "status", "--porcelain", "--", *CODE_DIRS], text=True
    ).strip()
    return f"{head}-dirty" if status else head
