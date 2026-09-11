from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-checkout config: {value}")
    return path


def inspect_prime_rl(recipe: dict, checkout: Path | None, result: dict) -> dict:
    """Inspect a pinned checkout; never install dependencies or execute training."""
    result["execution_scope"] = "Pinned upstream integration plan; GPU execution is not enabled"
    result["blockers"].append("GPU launch and spending controls are not implemented in this milestone")
    if platform.system() != "Linux":
        result["blockers"].append("This recipe targets a Linux CUDA host")
    result["required_gpus"] = recipe["required_gpus"]
    result["nvidia_smi_present"] = shutil.which("nvidia-smi") is not None
    if checkout is None:
        result["blockers"].append("Pass --checkout pointing to the pinned Prime-RL source")
        return result
    checkout = checkout.resolve()
    git = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    if git.returncode:
        result["blockers"].append("Checkout is not a readable Git repository")
        return result
    actual_revision = git.stdout.strip()
    result["checkout"] = str(checkout)
    result["actual_revision"] = actual_revision
    if actual_revision != recipe["revision"]:
        result["blockers"].append("Checkout revision does not match recipe pin")
    status = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, timeout=10, check=True)
    result["tracked_changes"] = bool(status.stdout.strip())
    if result["tracked_changes"]:
        result["blockers"].append("Checkout contains tracked changes")
    config = inside(checkout, recipe["config"])
    result["config_sha256"] = sha256(config)
    with config.open("rb") as stream:
        result["native_config"] = tomllib.load(stream)
    submodules = subprocess.run(["git", "-C", str(checkout), "submodule", "status"],
                                capture_output=True, text=True, timeout=10, check=True)
    result["submodules"] = submodules.stdout.splitlines()
    if any(line.startswith(("-", "+", "U")) for line in result["submodules"]):
        result["blockers"].append("Submodules are missing or differ from their pinned revisions")
    if not (checkout / ".venv/bin/python").exists():
        result["blockers"].append("Pinned backend environment has not been installed")
    result["native_dry_run_command"] = ["uv", "run", "--frozen", "--no-sync", "rl", "@",
                                         str(config), "--dry-run"]
    result["next_gate"] = "Install pins on an authorized CUDA host; validate native config, update/save/reload and evaluation"
    return result
