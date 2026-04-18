"""
Multi-node synchronization barrier using PyTorch's TCPStore.

TCPStore is PyTorch's built-in distributed key-value store -- the same
mechanism torchrun uses internally for node rendezvous. We use it here
to coordinate pipeline stages between DGX Sparks without timing assumptions.

Both ranks call the barrier with the same key. The barrier blocks until
both ranks have arrived, then both proceed simultaneously.

How it works:
    1. Rank 0 creates a TCPStore server; rank 1 connects as client
       (client retries automatically if server isn't up yet)
    2. Each rank sets its arrival key: "{key}_rank{rank}" = "1"
    3. Each rank calls wait() for both arrival keys
    4. wait() blocks (no polling) until both keys exist, then returns
    5. Both ranks proceed

Why TCPStore over raw sockets, sentinel files, or inotify:
    - Purpose-built for distributed training coordination
    - Blocking wait() with timeout (no polling, no filesystem watching)
    - Client auto-retries connection (no timing assumptions)
    - Already available in PyTorch (our hard dependency)
    - Battle-tested by every torchrun invocation worldwide

Usage (from speedrun_spark_bundle.sh):
    python3 runs/sync_barrier.py --rank 0 --addr 169.254.51.197 --key checkpoint_synced
    python3 runs/sync_barrier.py --rank 1 --addr 169.254.51.197 --key checkpoint_synced
"""

import argparse
import sys
from datetime import timedelta

import torch.distributed


def barrier(rank: int, addr: str, key: str, port: int = 29501, timeout: int = 600) -> None:
    """Block until both ranks reach this point.

    Args:
        rank: 0 for master (TCPStore server), 1 for client.
        addr: Master node's stacking cable IP address.
        key: Name for this synchronization point (e.g., "checkpoint_synced").
        port: TCP port for the store (default 29501, avoids torchrun's 29500).
        timeout: Seconds to wait before raising an error.
    """
    store = torch.distributed.TCPStore(
        host_name=addr,
        port=port,
        world_size=2,
        is_master=(rank == 0),
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout),
    )
    # Signal that this rank has arrived at the barrier
    store.set(f"{key}_rank{rank}", "1")
    # Block until both ranks have arrived
    store.wait(
        [f"{key}_rank0", f"{key}_rank1"],
        timedelta(seconds=timeout),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-node synchronization barrier using PyTorch TCPStore. "
        "Both ranks call this with the same --key. Blocks until both arrive."
    )
    p.add_argument("--rank", type=int, required=True, help="0 for master node, 1 for the other")
    p.add_argument("--addr", type=str, required=True, help="Master node stacking cable IP")
    p.add_argument("--key", type=str, required=True, help="Barrier name (e.g., checkpoint_synced)")
    p.add_argument(
        "--port",
        type=int,
        default=29501,
        help="TCPStore port (default: 29501, avoids torchrun's 29500)",
    )
    p.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    args = p.parse_args()

    try:
        barrier(args.rank, args.addr, args.key, args.port, args.timeout)
    except Exception as e:
        print(f"Barrier failed (rank {args.rank}, key={args.key}): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
