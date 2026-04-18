"""Tests for the PyTorch TCPStore synchronization barrier.

Tests the multi-node barrier used to coordinate pipeline stages between
two DGX Sparks. The barrier uses PyTorch's TCPStore -- the same mechanism
torchrun uses internally for node rendezvous.
"""

import os

# Add runs/ to path
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runs"))
from sync_barrier import barrier


def _run_barrier_in_thread(rank, addr, key, port, timeout=30, result=None):
    """Run barrier() in a thread, storing success/failure in result dict."""
    try:
        barrier(rank, addr, key, port=port, timeout=timeout)
        if result is not None:
            result[rank] = "ok"
    except Exception as e:
        if result is not None:
            result[rank] = f"error: {e}"


class TestBarrier:
    """Tests for the TCPStore barrier function."""

    def test_both_ranks_complete(self):
        """Both ranks arrive at barrier and proceed."""
        result = {}
        t0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "test1", 29601),
            kwargs={"result": result},
        )
        t1 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(1, "127.0.0.1", "test1", 29601),
            kwargs={"result": result},
        )
        t0.start()
        t1.start()
        t0.join(timeout=30)
        t1.join(timeout=30)
        assert result.get(0) == "ok", f"Rank 0: {result.get(0)}"
        assert result.get(1) == "ok", f"Rank 1: {result.get(1)}"

    def test_rank0_arrives_first(self):
        """Rank 0 starts first, rank 1 joins 2 seconds later."""
        result = {}
        t0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "test2", 29602),
            kwargs={"result": result},
        )
        t0.start()
        time.sleep(2)
        t1 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(1, "127.0.0.1", "test2", 29602),
            kwargs={"result": result},
        )
        t1.start()
        t0.join(timeout=30)
        t1.join(timeout=30)
        assert result.get(0) == "ok"
        assert result.get(1) == "ok"

    def test_rank1_arrives_first(self):
        """Rank 1 starts first, rank 0 joins 2 seconds later."""
        result = {}
        t1 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(1, "127.0.0.1", "test3", 29603),
            kwargs={"result": result},
        )
        t1.start()
        time.sleep(2)
        t0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "test3", 29603),
            kwargs={"result": result},
        )
        t0.start()
        t0.join(timeout=30)
        t1.join(timeout=30)
        assert result.get(0) == "ok"
        assert result.get(1) == "ok"

    def test_different_keys_dont_interfere(self):
        """Two barriers with different keys on different ports are independent."""
        result_a = {}
        result_b = {}
        # Barrier A
        ta0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "barrier_a", 29604),
            kwargs={"result": result_a},
        )
        ta1 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(1, "127.0.0.1", "barrier_a", 29604),
            kwargs={"result": result_a},
        )
        # Barrier B
        tb0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "barrier_b", 29605),
            kwargs={"result": result_b},
        )
        tb1 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(1, "127.0.0.1", "barrier_b", 29605),
            kwargs={"result": result_b},
        )
        ta0.start()
        ta1.start()
        tb0.start()
        tb1.start()
        ta0.join(timeout=30)
        ta1.join(timeout=30)
        tb0.join(timeout=30)
        tb1.join(timeout=30)
        assert result_a.get(0) == "ok"
        assert result_a.get(1) == "ok"
        assert result_b.get(0) == "ok"
        assert result_b.get(1) == "ok"

    @pytest.mark.slow
    def test_timeout_when_rank1_missing(self):
        """If only rank 0 arrives, barrier should timeout."""
        result = {}
        t0 = threading.Thread(
            target=_run_barrier_in_thread,
            args=(0, "127.0.0.1", "test_timeout", 29606),
            kwargs={"timeout": 5, "result": result},
        )
        t0.start()
        t0.join(timeout=15)
        assert "error" in result.get(0, ""), f"Expected timeout error, got: {result.get(0)}"
