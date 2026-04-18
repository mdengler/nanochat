"""Checkpoint validation utilities for the DGX Spark Bundle speedrun script.

PyTorch saves checkpoints as zip files (torch.save uses Python's zipfile module
internally). If power fails during a save, the file can be truncated, leaving a
corrupted zip. This module validates checkpoints and finds the latest valid one,
falling back to earlier checkpoints if the latest is corrupt.

Usage from speedrun_spark_bundle.sh:
    python3 runs/checkpoint_utils.py /path/to/base_checkpoints/d24

Usage from Python:
    from runs.checkpoint_utils import find_latest_valid_checkpoint
    step, path = find_latest_valid_checkpoint("/path/to/checkpoints")
"""

import glob
import os
import re
import sys
import zipfile


def validate_checkpoint(path: str) -> bool:
    """Return True if the checkpoint file is a valid zip (PyTorch format).

    PyTorch's torch.save writes files as zip archives. A truncated write
    (e.g., power failure during save) produces an invalid zip that would
    crash torch.load with RuntimeError or UnpicklingError.

    Args:
        path: Path to a .pt checkpoint file.

    Returns:
        True if the file is a valid, complete zip archive.
    """
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, FileNotFoundError, OSError):
        return False


def find_latest_valid_checkpoint(checkpoint_dir: str) -> tuple[int | None, str | None]:
    """Find the latest valid checkpoint, skipping corrupted ones.

    Scans the checkpoint directory for model_*.pt files, validates each
    starting from the highest step number. Corrupted files are renamed
    to .corrupt so they don't block future runs.

    Args:
        checkpoint_dir: Directory containing model_NNNNNN.pt files.

    Returns:
        (step, path) tuple for the latest valid checkpoint, or
        (None, None) if no valid checkpoint exists.
    """
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None, None

    # Sort by step number descending (try latest first)
    def step_from_path(p: str) -> int:
        match = re.search(r"model_(\d+)\.pt$", p)
        return int(match.group(1)) if match else -1

    files.sort(key=step_from_path, reverse=True)

    for f in files:
        if validate_checkpoint(f):
            return step_from_path(f), f
        corrupt_path = f + ".corrupt"
        print(f"WARNING: Corrupted checkpoint detected: {f} (renaming to {corrupt_path})")
        os.rename(f, corrupt_path)

    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    step, path = find_latest_valid_checkpoint(checkpoint_dir)
    if step is not None:
        print(step)
    # Exit 0 regardless -- caller checks stdout for step number
