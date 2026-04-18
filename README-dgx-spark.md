# Training nanochat on a DGX Spark Bundle

This document describes how to train a GPT-2 grade chatbot on an NVIDIA DGX Spark Bundle - two desktop-class Blackwell GPUs connected by a stacking cable. It is the homelab equivalent of Karpathy's [runs/speedrun.sh](runs/speedrun.sh), which targets cloud 8xH100 nodes.

Tested on DGX OS 7.4.0-7.5.0 with NGC container `nvcr.io/nvidia/pytorch:26.03-py3` (PyTorch 2.11, CUDA 13.2, NCCL 2.29.7, flash-attn 2.7.4).

All results in this document were measured against nanochat commit [`a445144`](https://github.com/karpathy/nanochat/commit/a445144) (2026-03-26).

## DGX Spark vs 8xH100

|                   | 8xH100 (cloud)       | DGX Spark Bundle              |
|-------------------|----------------------|-------------------------------|
| **GPU**           | 8x H100 SXM (Hopper) | 2x GB10 (Blackwell)           |
| **GPU memory**    | 8x 80GB = 640GB HBM3 | 2x 128GB = 256GB unified      |
| **Interconnect**  | NVLink 900GB/s       | 200Gbps RoCE (stacking cable) |
| **Attention**     | Flash Attention 3    | Flash Attention 2             |
| **Pretrain time** | ~1.65 hours          | 62.6 hours                    |
| **Full pipeline** | ~3 hours             | **3.0 days** (measured)       |
| **Cost per run**  | ~$72 (cloud rental)  | ~$0.50 (electricity)          |
| **Final val bpb** | ~0.72                | **0.720** (measured)          |
| **Model quality** | Identical            | Identical                     |

Same code, same model architecture, same training data, same CORE metric at the end. The only difference is wall clock time.

## Results

Full pipeline measured on a DGX Spark Bundle (2x GB10, RDMA stacking cable, NGC container, depth=24, 1.384B params, FA2+SSSL+FP8):

| Stage                | Duration                 | Result                  |
|----------------------|--------------------------|-------------------------|
| Dataset download     | 10 min                   | 171 shards, 15.7 GB     |
| Tokenizer training   | 14 min                   | 32,768 vocab BPE        |
| **Pretraining**      | **62.6 hours**           | **val bpb 0.720**       |
| Base eval (22 tasks) | 2.5 hours                | CORE 0.2646             |
| SFT                  | 5.9 hours                | 485 steps, loss 0.84    |
| Chat eval (6 tasks)  | 1.8 hours                | ChatCORE 0.3599         |
| **Total pipeline**   | **~73 hours (3.0 days)** | **GPT-2 grade chatbot** |

Training throughput: 25,900 tok/sec sustained, 29.6% MFU, 5,568 steps, 5.84B tokens.

## Prerequisites

### Hardware

1. **Two DGX Sparks** connected via the QSFP/CX7 stacking cable (included in the Bundle).
2. **Stacking cable configured** - follow NVIDIA's official playbooks:
   - [connect-two-sparks](https://build.nvidia.com/spark/connect-two-sparks) - configures link-local networking (netplan), passwordless SSH over the cable, and the `discover-sparks` script
   - [NCCL for stacked Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks) - verifies NCCL can communicate over the stacking cable

   These playbooks set up everything automatically: SSH keys (`~/.ssh/id_ed25519_shared`), netplan config (`/etc/netplan/40-cx7.yaml`), and link-local IP addressing. Every DGX Spark gets a unique link-local IP derived from its MAC address.

### Why NCCL?

[NCCL](https://developer.nvidia.com/nccl) (NVIDIA Collective Communications Library) is how the two GPUs synchronize gradients during distributed training. Without it, each GPU would train on its own data but never share what it learned. NCCL uses the 200Gbps stacking cable (via RDMA/RoCE) to keep both GPUs in sync - every training step, they average their gradients so the model converges as if it were training on one big GPU.

### Software

1. **NGC container** pulled on both Sparks: `docker pull nvcr.io/nvidia/pytorch:26.03-py3`
2. **nanochat cloned** to `~/nanochat` on both Sparks, same commit.
3. **(Optional)** Store a HuggingFace token for faster SFT dataset downloads. Without it, downloads work but with rate-limit warnings. Run on both Sparks (`read -s` hides input so the token never appears in terminal history):
   ```bash
   read -sp "HuggingFace token: " HF_TOKEN && echo
   mkdir -p ~/.cache/huggingface
   echo "$HF_TOKEN" > ~/.cache/huggingface/token
   chmod 600 ~/.cache/huggingface/token
   unset HF_TOKEN
   ```

### System configuration

**Disable auto-suspend.** Ubuntu's GNOME desktop suspends the machine after idle timeout. This will kill a multi-day training run. Disable it on **both** Sparks:

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

**Find your stacking cable IPs.** Each DGX Spark gets a unique link-local IP. You need yours for `MASTER_ADDR`:

```bash
# On each Spark:
ip addr show enp1s0f1np1 | grep "inet "
# Example output: inet 169.254.51.197/16 ...
#                      ^^^^^^^^^^^^^^ this is your IP
```

If your network interface has a different name than `enp1s0f1np1`, check with `ibdev2netdev` and set `NCCL_IFNAME` accordingly.

**Verify the stacking cable:**

```bash
ibdev2netdev | grep Up                          # should show your stacking interface
ethtool enp1s0f1np1 | grep Speed                # should show 200000Mb/s
ping -c 3 -I enp1s0f1np1 <other-spark-IP>       # should show <2ms latency
```

## Quick start

Open two terminal tabs (one SSH session per Spark). On each Spark, start the NGC
container. The `--ulimit` flags are [recommended by NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
(also printed in the NGC container startup banner). `--cap-add SYS_RESOURCE` allows
HumanEval's code sandbox to set resource limits (`nanochat/execution.py:152`) --
without it, HumanEval scores 0%:

```bash
docker run --gpus all -it --rm \
    --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_RESOURCE \
    --device /dev/infiniband \
    -v ~/nanochat:/workspace/nanochat \
    -v ~/.cache/nanochat:/root/.cache/nanochat \
    -v ~/.cache/torchinductor:/tmp/torchinductor_root \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.ssh:/root/.ssh:ro \
    nvcr.io/nvidia/pytorch:26.03-py3 bash
```

Inside each container, install nanochat (preserving the container's CUDA PyTorch):

```bash
cd /workspace/nanochat
pip install --no-deps -e .
pip install fastapi kernels rustbpe tiktoken uvicorn wandb
```

Then run the speedrun on both Sparks simultaneously:

```bash
# On spark A (rank 0, master):
NODE_RANK=0 bash runs/speedrun_spark_bundle.sh

# On spark B (rank 1):
NODE_RANK=1 bash runs/speedrun_spark_bundle.sh
```

Come back in ~3 days. If it crashes (power outage, network glitch, etc.), just
re-run the same commands on both Sparks - the script auto-detects the latest
checkpoint and resumes. Checkpoints are saved every 250 steps (~2.8 hours).

Chat with your model:

```bash
python -m scripts.chat_cli -p "Why is the sky blue?"
```

## What we optimized

We ran a systematic optimization study to squeeze the most out of the DGX Spark Bundle. Here's what we found, in order of impact:

### 1. NGC container is essential (5x speedup)

Native PyTorch 2.9.1 runs at 21K tok/sec on depth=8. The NGC container (PyTorch 2.11) runs at 105K - a **5x improvement**. Root cause: native PyTorch lacks sm_121a (Blackwell) kernels, forcing slow PTX JIT compilation. The NGC container ships pre-compiled kernels. Without it, the Sparks are unusable for serious training.

### 2. Flash Attention 2 unlocks SSSL (28% faster at depth=24)

Karpathy uses Flash Attention 3 on Hopper (H100), which supports sliding window attention (SSSL pattern). FA3 will never work on GB10 - it requires Hopper's TMA (Tensor Memory Accelerator) and WGMMA instructions, which are silicon features not present on Blackwell.

However, the NGC container ships **flash-attn 2.7.4**, which works on GB10. We added FA2 as a middle tier in nanochat's attention backend: FA3 (Hopper) > FA2 (flash-attn package) > SDPA (any device). FA2 provides native sliding window kernels, unlocking Karpathy's SSSL window pattern on GB10.

Results on depth=24: FA2+SSSL = 11,760 tok/sec vs SDPA+L = 9,220 tok/sec (**28% faster**). FA2 also revealed that the previous "compute ceiling" of 10,400 tok/sec was actually an SDPA bottleneck, not a hardware limit.

### 3. FP8 training (12% faster on large models)

FP8 quantization speeds up matmuls on Blackwell. At depth=24 (n_embd=1536), FP8 is 12% faster than BF16 on top of FA2. At depth=8 (n_embd=512), FP8 is actually 14% *slower* - the quantization overhead dominates for small matrices. FP8 also reduces memory usage (51.4 GB vs 58.7 GB at depth=24 batch=16).

### 4. RDMA over stacking cable (near-perfect 2x scaling)

The QSFP/CX7 stacking cable provides 200Gbps RDMA/RoCE between the two Sparks. With proper NCCL configuration (16 channels, IBext_v11 transport), we achieve 1.96x scaling at depth=24 (97.8% per-GPU efficiency) - the remaining 2.2% overhead is gradient synchronization. Communication is NOT the bottleneck.

### 5. torch.compile cache persistence (3.2x faster restarts)

The NGC container's torch.compile generates optimized kernels on the first training step (~23s for depth=8, ~120-165s for depth=24). These are stored in `/tmp/torchinductor_root/` inside the container and lost when the container exits.

Mounting this directory as a host volume (`-v ~/.cache/torchinductor:/tmp/torchinductor_root`) persists the cache across container restarts. Subsequent starts are 3.2x faster. This matters for homelab users who restart containers frequently.

### 6. Eval warmup (prevents crash on UMA systems)

During validation eval, `disable_fp8()` swaps FP8 modules to BF16 inside the compiled model. This invalidates torch.compile's cached graph and forces recompilation. On UMA systems with training memory already allocated, the recompilation memory spike can exceed physical DRAM and crash the machine.

Fix: pre-compile the BF16 eval graph during the warmup phase, before the training loop allocates its full memory. This is included in the code - no action needed from users.

## Stage-by-stage timing

The speedrun script runs 6 stages in this order. Stages 1-2 run on both Sparks independently (each downloads its own data and trains its own tokenizer - both produce identical results). Stages 3 and 4 are distributed (both Sparks coordinate via NCCL). Stages 5 and 6 run on rank 0 only.

| Stage           | What                                            | Duration        | Both Sparks?   | Notes                                         |
|-----------------|-------------------------------------------------|-----------------|----------------|-----------------------------------------------|
| 1. dataset      | Download ~170 data shards                       | ~10 min         | Yes            | Idempotent, each Spark downloads its own copy |
| 2. tokenizer    | Train BPE tokenizer (vocab=32768)               | ~14 min         | Yes            | Deterministic, identical on both              |
| 3. **pretrain** | **Train depth=24 model (~5.84B tokens)**        | **~62.6 hours** | **Yes (NCCL)** | **Distributed, auto-resumable**               |
| 4. sft          | Supervised fine-tuning (conversation, tool use) | ~5.9 hours      | Yes (NCCL)     | Checkpoint synced from rank 0 to rank 1       |
| 5. eval         | CORE metric evaluation (22 tasks)               | ~2.5 hours      | Rank 0 only    |                                               |
| 6. chateval     | Chat model evaluation (6 tasks)                 | ~1.8 hours      | Rank 0 only    |                                               |

**Why does each Spark need its own data?** In distributed training, each GPU reads different portions of the dataset in parallel (DDP sharding). The dataloader on each Spark reads from its local filesystem - there's no network filesystem sharing. Both Sparks download the same shards, but each reads different row groups during training.

**Running individual stages:** `STAGE=pretrain NODE_RANK=0 bash runs/speedrun_spark_bundle.sh`

**Stopping and resuming:** If pretraining is interrupted (crash, power outage, etc.), just re-run the same command - the script auto-detects the latest checkpoint and resumes. You can also specify a step explicitly with `RESUME_STEP=500`. Between stages 3 and 4, the script automatically copies the pretrain checkpoint from rank 0 to rank 1 via SSH over the stacking cable.

## Known limitations

- **No Flash Attention 3.** FA3 requires Hopper's TMA and WGMMA instructions - silicon features not present on Blackwell. No software update will fix it. FA2 is sufficient.
- **No FP4 training.** Blackwell FP4 GEMM kernels overflow GB10's shared memory (99 KiB vs B200's 228 KiB). FP8 is the best low-precision option for training.
- **RoCE, not NVLink.** The 200 Gbps stacking cable (25 GB/s) is 36x slower than H100's NVLink (900 GB/s). Despite this, gradient sync is not the bottleneck (97.8% per-GPU efficiency).
- **2 GPUs max with cable.** Point-to-point stacking supports 2 Sparks. 3-4 Sparks require a RoCE switch ([docs](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)).
- **NGC container required.** Native PyTorch lacks sm_121a kernel support. Always use the NGC container for training.
- **Batch size 16, not 32.** Batch=32 uses 94 GB of 128 GB unified memory, leaving insufficient headroom for OS operations. Batch=16 uses 56 GB (44%) with negligible throughput difference.

## Single Spark

If you have only one DGX Spark (no stacking cable), run setup stages then train with a single GPU:

```bash
# Inside NGC container:
python -m nanochat.dataset -n 170
python -m scripts.tok_train
python -m scripts.tok_eval

# Pretrain (single GPU, no torchrun):
python -m scripts.base_train \
    --depth=24 --run=dummy \
    --device-batch-size=16 --window-pattern=SSSL --fp8 \
    --save-every=250 --core-metric-every=999999 --sample-every=-1

# Expected: ~13,125 tok/sec, ~5.2 days
```

## Configuration reference

All configuration is via environment variables (override defaults without editing the script):

| Variable            | Default          | Description                                                                                |
|---------------------|------------------|--------------------------------------------------------------------------------------------|
| `NODE_RANK`         | (required)       | `0` for master Spark, `1` for the other                                                    |
| `STAGE`             | `all`            | Run specific stage: `dataset`, `tokenizer`, `pretrain`, `sft`, `eval`, `chateval`          |
| `RESUME_STEP`       | (auto-detect)    | Resume from this step (auto-detects latest checkpoint if not set)                          |
| `WANDB_RUN`         | `dummy`          | wandb run name (`dummy` disables logging)                                                  |
| `DEPTH`             | `24`             | Transformer depth (12 for quick experiments, 24-26 for GPT-2)                              |
| `DEVICE_BATCH_SIZE` | `16`             | Per-GPU batch size (16 is safe for 128 GB UMA; 32 risks OOM)                               |
| `NUM_ITERATIONS`    | `-1` (auto)      | Explicit step count for pretraining (`-1` = compute-optimal)                               |
| `MASTER_ADDR`       | `169.254.51.197` | Stacking cable IP of the master Spark (rank 0). Find yours with `ip addr show enp1s0f1np1` |
| `RANK1_ADDR`        | `169.254.123.70` | Stacking cable IP of rank 1. Find yours with `ip addr show enp1s0f1np1`                    |
| `REMOTE_USER`       | `$USER`          | Username for SSH between Sparks (checkpoint sync)                                          |
| `NCCL_IFNAME`       | `enp1s0f1np1`    | Stacking cable network interface                                                           |
