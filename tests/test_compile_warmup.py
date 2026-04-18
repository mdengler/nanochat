"""Tests for the multi-node compile warmup + barrier logic.

Verifies that the warmup code (added to base_train.py and chat_sft.py)
correctly gates on ddp_world_size > 1 and calls dist.barrier() after
the dummy forward+backward pass.

These tests use grep to verify the code structure rather than executing
the training scripts (which would require a full DDP setup).
"""

import os

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read_file(relpath):
    """Read a file relative to the repo root."""
    path = os.path.join(REPO_ROOT, relpath)
    with open(path) as f:
        return f.read()


class TestWarmupCodePresence:
    """Verify the warmup+barrier code is correctly placed in training scripts."""

    def test_base_train_has_warmup(self):
        code = _read_file("scripts/base_train.py")
        assert "if ddp and ddp_world_size > 1:" in code
        assert "dist.barrier()" in code
        assert "Multi-node compile warmup complete" in code

    def test_chat_sft_has_warmup(self):
        code = _read_file("scripts/chat_sft.py")
        assert "if ddp and ddp_world_size > 1:" in code
        assert "dist.barrier()" in code
        assert "Multi-node compile warmup complete" in code

    def test_warmup_after_torch_compile(self):
        """Warmup must come AFTER torch.compile(), not before."""
        code = _read_file("scripts/base_train.py")
        compile_pos = code.find("torch.compile(model")
        warmup_pos = code.find("Multi-node compile warmup")
        assert compile_pos > 0, "torch.compile not found"
        assert warmup_pos > 0, "warmup not found"
        assert warmup_pos > compile_pos, "warmup must come after torch.compile"

    def test_warmup_cleans_up(self):
        """Warmup should clean up dummy tensors and cache."""
        code = _read_file("scripts/base_train.py")
        assert "del warmup_x, warmup_y" in code
        assert "torch.cuda.empty_cache()" in code

    def test_warmup_includes_eval_graph(self):
        """Warmup should also compile the BF16 eval graph (disable_fp8 path)."""
        code = _read_file("scripts/base_train.py")
        warmup_pos = code.find("Multi-node compile warmup")
        assert warmup_pos > 0
        # The eval warmup must happen before the barrier
        barrier_pos = code.find("dist.barrier()", warmup_pos - 500)
        disable_fp8_pos = code.find("disable_fp8(model)", warmup_pos - 500)
        assert disable_fp8_pos > 0, "eval warmup (disable_fp8) not found near compile warmup"
        assert disable_fp8_pos < barrier_pos, "eval warmup must come before dist.barrier()"

    def test_warmup_uses_model_config(self):
        """Warmup should use model_config for vocab_size and sequence_len."""
        code = _read_file("scripts/base_train.py")
        assert "model_config.vocab_size" in code
        assert "model_config.sequence_len" in code


class TestMfuInit:
    """Tests for the mfu = 0 initialization fix."""

    def test_mfu_defined_outside_if_else(self):
        """mfu must be defined outside the if/else resuming block."""
        code = _read_file("scripts/base_train.py")
        # Find the if/else block and the mfu = 0 line
        lines = code.split("\n")
        mfu_line = None
        in_if_block = False
        in_else_block = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "if not resuming:":
                in_if_block = True
            elif stripped == "else:" and in_if_block:
                in_else_block = True
                in_if_block = False
            elif in_else_block and not line.startswith("    ") and stripped:
                in_else_block = False
            if stripped == "mfu = 0":
                mfu_line = i
                break
        assert mfu_line is not None, "mfu = 0 not found"
        # mfu = 0 should NOT be indented inside if or else
        assert not lines[mfu_line].startswith("    "), (
            "mfu = 0 should be at module level, not inside if/else"
        )

    def test_mfu_has_comment(self):
        """mfu initialization should have a comment explaining why."""
        code = _read_file("scripts/base_train.py")
        assert "--resume-from-step == num_iterations" in code


class TestFAStatusMessages:
    """Tests for the FA2 status message additions."""

    def test_base_train_fa2_message(self):
        code = _read_file("scripts/base_train.py")
        assert "Flash Attention 2" in code
        assert "USE_FA2" in code

    def test_base_train_preserves_fa3_message(self):
        """Karpathy's original FA3 message must be preserved verbatim."""
        code = _read_file("scripts/base_train.py")
        assert "Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome." in code

    def test_chat_sft_fa2_condition(self):
        code = _read_file("scripts/chat_sft.py")
        assert "not HAS_FA3 and not HAS_FA2" in code

    def test_chat_sft_preserves_message_structure(self):
        """Karpathy's warning message structure must be preserved."""
        code = _read_file("scripts/chat_sft.py")
        assert "Flash Attention 3 not available, using PyTorch SDPA fallback" in code
