"""
Generate synthetic Q&A pairs from non-primary corpus documents using the base model.

Run once before chat_sft.py:
    python -m scripts.generate_domain_qa --model-tag d26 --max-docs 5000 --qa-per-doc 2

Outputs to {base_dir}/domain_qa/{corpus_name}.jsonl in CustomJSON format:
    [{"role":"user","content":"<question>"},{"role":"assistant","content":"<answer>"}]
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.dataset import get_corpuses, parquets_iter_batched
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Generate domain Q&A pairs from non-primary corpus")
parser.add_argument("--model-tag", type=str, default=None, help="base checkpoint tag (default: auto-detect largest)")
parser.add_argument("--model-step", type=int, default=None, help="checkpoint step (default: latest)")
parser.add_argument("--max-docs", type=int, default=5000, help="max documents per corpus to process")
parser.add_argument("--qa-per-doc", type=int, default=2, help="Q&A pairs to generate per document chunk")
parser.add_argument("--chunk-chars", type=int, default=800, help="document excerpt length in chars")
parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing JSONL files")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Initialize compute
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Only rank 0 generates; exit early for other ranks in DDP
if ddp_rank != 0:
    import torch.distributed as dist
    dist.barrier()
    dist.destroy_process_group()
    exit(0)

# -----------------------------------------------------------------------------
# Load model
model, tokenizer, meta = load_model("base", device, phase="eval",
                                     model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)
bos_token_id = tokenizer.get_bos_token_id()

# -----------------------------------------------------------------------------
# Prepare output directory
base_dir = get_base_dir()
qa_dir = os.path.join(base_dir, "domain_qa")
os.makedirs(qa_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Few-shot prompt template (hard-coded 2 examples)
FEW_SHOT_PREFIX = """\
Generate a question and detailed answer based on the following passage.

### Passage:
Photosynthesis is the process by which plants use sunlight, water, and CO\u2082 \
to produce sugar and oxygen. Chlorophyll in plant cells absorbs light energy.

### Question:
What role does chlorophyll play in photosynthesis?

### Answer:
Chlorophyll is the pigment in plant cells that absorbs light energy, which \
is then used to convert water and CO\u2082 into sugar and oxygen.

---

### Passage:
The Magna Carta, signed in 1215, was a royal charter of rights agreed to by \
King John of England. It established for the first time that the king was \
subject to the rule of law and protected the rights of barons and the church.

### Question:
What was the historical significance of the Magna Carta?

### Answer:
The Magna Carta was historically significant because it established the \
principle that the monarch was subject to the rule of law, limiting royal \
power. Signed in 1215, it laid the groundwork for constitutional governance \
and the protection of individual rights.

---

### Passage:
{chunk}

### Question:
"""

def extract_chunk(text, chunk_chars):
    """Extract a chunk from the middle of the document to avoid header/footer boilerplate."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return text
    mid = len(text) // 2
    start = max(0, mid - chunk_chars // 2)
    return text[start:start + chunk_chars]

def parse_qa(completion_text):
    """
    Parse the generated completion to extract Question and Answer strings.
    Expected format:
        <question text>

        ### Answer:
        <answer text>
    Returns (question, answer) or (None, None) if parsing fails.
    """
    # The completion starts right after "### Question:\n" in the prompt
    # so the completion is: "<question>\n\n### Answer:\n<answer>"
    if "### Answer:" not in completion_text:
        return None, None
    parts = completion_text.split("### Answer:", 1)
    question = parts[0].strip()
    answer = parts[1].strip()
    # Cut off at next section marker if present
    for marker in ["---", "### Passage:", "### Question:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    return question, answer

# -----------------------------------------------------------------------------
# Process each non-primary corpus
corpuses = get_corpuses()
domain_corpuses = {name: paths for name, paths in corpuses.items() if name != "primary"}

if not domain_corpuses:
    print0("No non-primary corpora found. Nothing to generate.")
    exit(0)

print0(f"Found {len(domain_corpuses)} non-primary corpus/corpora: {list(domain_corpuses.keys())}")

for corpus_name, parquet_paths in sorted(domain_corpuses.items()):
    output_path = os.path.join(qa_dir, f"{corpus_name}.jsonl")
    if os.path.exists(output_path) and not args.overwrite:
        print0(f"Skipping {corpus_name}: {output_path} already exists (use --overwrite to regenerate)")
        continue

    print0(f"\nProcessing corpus '{corpus_name}' ({len(parquet_paths)} files)...")
    n_generated = 0
    n_filtered = 0
    n_docs_processed = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for filepath in parquet_paths:
            if n_docs_processed >= args.max_docs:
                break
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                if n_docs_processed >= args.max_docs:
                    break
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for text in texts:
                    if n_docs_processed >= args.max_docs:
                        break
                    chunk = extract_chunk(text, args.chunk_chars)
                    if len(chunk) < 100:  # skip very short chunks
                        continue
                    prompt_text = FEW_SHOT_PREFIX.format(chunk=chunk)
                    prompt_tokens = tokenizer.encode(prompt_text, prepend=bos_token_id)

                    for _ in range(args.qa_per_doc):
                        try:
                            with autocast_ctx:
                                results, _ = engine.generate_batch(
                                    prompt_tokens,
                                    num_samples=1,
                                    max_tokens=256,
                                    temperature=args.temperature,
                                    seed=n_docs_processed * 100 + n_generated,
                                )
                            completion_tokens = results[0][len(prompt_tokens):]
                            completion_text = tokenizer.decode(completion_tokens)
                            question, answer = parse_qa(completion_text)

                            if (question is None or answer is None
                                    or len(question) < 10
                                    or len(answer) < 20):
                                n_filtered += 1
                                continue

                            record = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer},
                            ]
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            n_generated += 1
                        except Exception as e:
                            n_filtered += 1
                            continue

                    n_docs_processed += 1

    print0(f"  Corpus '{corpus_name}': {n_docs_processed} docs processed, "
           f"{n_generated} Q&A pairs written to {output_path}, "
           f"{n_filtered} filtered/skipped")

print0("\nDone. Run chat_sft.py with --domain-qa-epochs to include these pairs in SFT.")
