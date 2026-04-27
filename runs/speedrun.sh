#!/bin/bash

set -e
set -x

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

if [ $(whoami) != "root" ] ; then
# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
fi


# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROCS=$(python -c 'import torch ; print (0 if not torch.cuda.is_available() else torch.cuda.device_count())')
NPROC_PER_NODE=gpu

export MY_CUDA_VER=13
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvshmem/${MY_CUDA_VER}:${LD_LIBRARY_PATH}
export TRITON_PTXAS_PATH=/usr/local/cuda-${MY_CUDA_VER}/bin/ptxas
export CUDA_HOME=/usr/local/cuda-${MY_CUDA_VER}
export PATH=/usr/local/cuda-${MY_CUDA_VER}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${MY_CUDA_VER}/lib64:${LD_LIBRARY_PATH}

BATCH_SIZE=32
# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
DEPTH=24
PARAM_DATA_RATIO=8

WINDOW_PATTERN=
if [ "${NPROCS}" == 1 ] ; then
    DEPTH=4
    WINDOW_PATTERN="--window-pattern L"
    BATCH_SIZE=16
fi

# for comparison / benchmarking, run this script within the NVIDIA docker container:
####  docker pull nvcr.io/nvidia/pytorch:25.09-py3
####  # run below from within the nanochat directory
####  docker run --gpus all -it --rm --ipc=host \
####  -v $HOME/.cache/nanochat:/root/.cache/nanochat \
####  -v ${PWD}:/workspace -w /workspace \
####  nvcr.io/nvidia/pytorch:25.09-py3
####
####  # install inside the container: pandas pyarrow wandb tokenizers tiktoken
####  # run inside the nanochat directory inside the container: python -m scripts.base_train --depth=2

# Optional: domain corpus directories to blend into pretraining.
# Space-separated paths in CORPUS env var; each becomes a --corpus=<path> arg
# so base_train.py auto-prepares it via scripts.prepare_corpus before training.
# Usage: CORPUS="/data/legal /data/medical" bash runs/speedrun.sh
CORPUS_ARGS=""
if [ -n "$CORPUS" ]; then
    for dir in $CORPUS; do
        CORPUS_ARGS="$CORPUS_ARGS --corpus=$dir"
    done
fi

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=$DEPTH --target-param-data-ratio=$PARAM_DATA_RATIO --run=$WANDB_RUN --device-batch-size=${BATCH_SIZE} --fp8 --save-every=10000 --sample-every=100 ${WINDOW_PATTERN} ${CORPUS_ARGS}

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=${BATCH_SIZE}

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.generate_domain_qa --model-tag d${DEPTH}

# run SFT and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=${BATCH_SIZE} --run=$WANDB_RUN  --domain-qa-epochs=3
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
