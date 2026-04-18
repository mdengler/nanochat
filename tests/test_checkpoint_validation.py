"""Tests for checkpoint validation and corruption recovery.

These tests verify that the speedrun script correctly handles corrupted
checkpoint files (e.g., from power failure during torch.save) by falling
back to the previous valid checkpoint.
"""

import os

# Add runs/ to path so we can import checkpoint_utils
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runs"))
from checkpoint_utils import find_latest_valid_checkpoint, validate_checkpoint


def _create_valid_checkpoint(directory, step):
    """Create a valid .pt checkpoint file using torch.save."""
    path = os.path.join(directory, f"model_{step:06d}.pt")
    torch.save({"step": step, "weights": torch.randn(10)}, path)
    return path


def _create_corrupt_checkpoint(directory, step):
    """Create a truncated/corrupt .pt file (simulates power failure mid-save)."""
    path = os.path.join(directory, f"model_{step:06d}.pt")
    with open(path, "wb") as f:
        f.write(b"PK\x03\x04truncated garbage data")
    return path


def _create_empty_checkpoint(directory, step):
    """Create an empty 0-byte .pt file."""
    path = os.path.join(directory, f"model_{step:06d}.pt")
    open(path, "wb").close()
    return path


class TestValidateCheckpoint:
    """Tests for validate_checkpoint()."""

    def test_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = _create_valid_checkpoint(d, 100)
            assert validate_checkpoint(path) is True

    def test_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = _create_corrupt_checkpoint(d, 100)
            assert validate_checkpoint(path) is False

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = _create_empty_checkpoint(d, 100)
            assert validate_checkpoint(path) is False

    def test_nonexistent_file(self):
        assert validate_checkpoint("/nonexistent/path/model_000100.pt") is False

    def test_random_binary(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model_000100.pt")
            with open(path, "wb") as f:
                f.write(os.urandom(1024))
            assert validate_checkpoint(path) is False


class TestFindLatestValidCheckpoint:
    """Tests for find_latest_valid_checkpoint()."""

    def test_no_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            step, path = find_latest_valid_checkpoint(d)
            assert step is None
            assert path is None

    def test_single_valid(self):
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 250)
            step, path = find_latest_valid_checkpoint(d)
            assert step == 250
            assert "model_000250.pt" in path

    def test_multiple_valid_returns_highest(self):
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 100)
            _create_valid_checkpoint(d, 250)
            _create_valid_checkpoint(d, 500)
            step, path = find_latest_valid_checkpoint(d)
            assert step == 500

    def test_fallback_to_previous(self):
        """Latest checkpoint is corrupt, should fall back to previous valid one."""
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 250)
            _create_corrupt_checkpoint(d, 500)
            step, path = find_latest_valid_checkpoint(d)
            assert step == 250
            assert "model_000250.pt" in path

    def test_all_corrupt(self):
        with tempfile.TemporaryDirectory() as d:
            _create_corrupt_checkpoint(d, 250)
            _create_corrupt_checkpoint(d, 500)
            step, path = find_latest_valid_checkpoint(d)
            assert step is None
            assert path is None

    def test_corrupt_files_renamed(self):
        """Corrupted files should be renamed to .corrupt."""
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 250)
            corrupt_path = _create_corrupt_checkpoint(d, 500)
            find_latest_valid_checkpoint(d)
            assert not os.path.exists(corrupt_path)
            assert os.path.exists(corrupt_path + ".corrupt")

    def test_valid_files_not_renamed(self):
        """Valid files should NOT be renamed."""
        with tempfile.TemporaryDirectory() as d:
            valid_path = _create_valid_checkpoint(d, 250)
            find_latest_valid_checkpoint(d)
            assert os.path.exists(valid_path)
            assert not os.path.exists(valid_path + ".corrupt")

    def test_nonexistent_directory(self):
        step, path = find_latest_valid_checkpoint("/nonexistent/dir")
        assert step is None
        assert path is None

    def test_step_zero(self):
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 0)
            step, path = find_latest_valid_checkpoint(d)
            assert step == 0

    def test_mixed_corruption_pattern(self):
        """Steps 100 (valid), 200 (corrupt), 300 (corrupt), 400 (valid)
        should return 400 and rename nothing."""
        with tempfile.TemporaryDirectory() as d:
            _create_valid_checkpoint(d, 100)
            _create_corrupt_checkpoint(d, 200)
            _create_corrupt_checkpoint(d, 300)
            _create_valid_checkpoint(d, 400)
            step, path = find_latest_valid_checkpoint(d)
            assert step == 400
            # 200 and 300 should NOT be renamed (we stop at first valid)
            assert os.path.exists(os.path.join(d, "model_000200.pt"))
