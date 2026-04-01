# Plan: Corpus Blend + Domain Q&A SFT

## Context

nanochat pretrains on FineWeb-EDU parquet shards. Users want to add proprietary corpora
(legal docs, technical manuals, codebases, etc.) that the model should deeply memorise
alongside general language understanding. The goal is:

1. **Feature 1 – Corpus Blend**: Modify the pretraining dataloader to automatically
   oversample every non-primary corpus so it receives exactly **m=10 full epochs** during
   any training run (where primary = FineWeb-EDU files named `shard_*.parquet`).

2. **Feature 2 – Domain Q&A SFT**: A separate script uses the pretrained base model to
   generate synthetic Q&A pairs from non-primary corpus documents; those pairs are then
   folded into the existing `chat_sft.py` mixture for a short SFT pass.

---

## Key Design Decisions

### Feature 1 – m semantics
m=10 means **absolute epochs**: the domain corpus is consumed exactly 10 times during
the training run. Training is extended: `T' = T_FineWeb + m × S_domain`.

num_iterations is updated in base_train.py accordingly.

The oversampling is implemented by **repeating domain file paths** k times in the
scheduled file list, where:

```
k = round(m × S_primary / T_FineWeb)
```

Proof (verify in planning):
- Schedule cycle token count = S_primary + k × S_domain
- Cycles run = T' / cycle_size = (T_FW + m×S_d) / (S_p + k×S_d)
  = T_FW/S_p  (when k = m × S_p / T_FW)
- Primary tokens seen = cycles × S_p = T_FW ✓
- Domain tokens seen  = cycles × k × S_d = m × S_d ✓

S_primary and S_domain are **estimated from parquet file sizes** (bytes ÷ 4, since
English BPE ~4 bytes/token). Approximation is good enough for the ±25% variance in
shard token density; a precise count is not needed.

Multiple non-primary corpora each get their own k independently (same formula, same m).

### Feature 2 – Q&A generation flow
- Separate script `scripts/generate_domain_qa.py` (run once before chat_sft).
- Reads non-primary corpus parquets via `get_corpuses()`.
- Loads the base model (via existing `load_model("base", ...)`) and uses `Engine` for
  inference with a hard-coded few-shot prompt.
- Outputs to `{base_dir}/domain_qa/{corpus_name}.jsonl` in the existing CustomJSON format.
- `chat_sft.py` auto-detects files in that directory and adds them to the mixture.

---

## Phase 1: Common Foundation – `get_corpuses()`

**File**: `nanochat/dataset.py`

Add two functions after `list_parquet_files()`:

```python
def get_corpuses(data_dir=None):
    """
    Returns dict mapping corpus_name -> sorted list of parquet file paths.
    - Files named "shard_*.parquet" → corpus_name = "primary"
    - Files named "foo_*.parquet"   → corpus_name = "foo"
    The last file overall (alphabetically) is reserved for validation and excluded.
    """
    all_paths = list_parquet_files(data_dir)          # already sorted
    train_paths = all_paths[:-1]                       # exclude last file (val shard)
    corpuses = {}
    for path in train_paths:
        prefix = os.path.basename(path).split('_')[0]
        name = "primary" if prefix == "shard" else prefix
        corpuses.setdefault(name, []).append(path)
    return corpuses

def estimate_corpus_tokens(file_paths, bytes_per_token=4):
    """Estimate total token count from file sizes. Good to ±25%."""
    total_bytes = sum(os.path.getsize(p) for p in file_paths)
    return total_bytes // bytes_per_token
```

**Note**: `get_corpuses()` is shared by both features.

---

## Phase 2: Feature 1 – Corpus Blend in Dataloader

### 2a. Build the scheduled file list (`nanochat/dataset.py`)

Add after `get_corpuses()`:

```python
def build_blend_schedule(corpuses, m, target_primary_tokens):
    """
    Returns a flat list of parquet file paths where each non-primary corpus's
    files appear k times (k = round(m × S_primary / target_primary_tokens))
    interleaved evenly with primary files.

    Interleaving: np_total domain-file slots are distributed evenly across
    the primary slots using round-robin so domain data is spread throughout.

    Args:
        corpuses: dict from get_corpuses()
        m: oversampling factor (e.g. 10)
        target_primary_tokens: T_FineWeb (Chinchilla-optimal tokens for primary)

    Returns:
        scheduled: list[str] of file paths (primary once, each domain corpus k times)
        report: dict with per-corpus stats for logging
    """
    primary_paths = corpuses.get("primary", [])
    s_primary = estimate_corpus_tokens(primary_paths)
    k_factor = m * s_primary / max(target_primary_tokens, 1)

    domain_slots = []  # flat list of (path, corpus_name) to interleave
    report = {"primary": {"files": len(primary_paths), "est_tokens": s_primary, "repeats": 1}}
    # k is the same for ALL non-primary corpora. Derivation:
    # We want each domain token seen m times while primary is seen T_FW/S_primary fraction.
    # Cycles through schedule = T_FW / S_primary
    # Domain tokens per cycle = k × S_domain
    # Total domain tokens = cycles × k × S_domain = m × S_domain  →  k = m × S_p / T_FW
    k = max(1, round(m * s_primary / max(target_primary_tokens, 1)))

    for name, paths in corpuses.items():
        if name == "primary":
            continue
        s_domain = estimate_corpus_tokens(paths)
        domain_slots.extend(paths * k)
        # effective_epochs ≈ m (guaranteed by k formula)
        report[name] = {"files": len(paths), "est_tokens": s_domain,
                        "repeats": k, "effective_epochs": m}

    if not domain_slots:
        return primary_paths, report

    # Evenly interleave domain slots across primary slots
    result = []
    n_p, n_d = len(primary_paths), len(domain_slots)
    ratio = n_d / max(n_p, 1)
    d_cursor = 0
    for i, pf in enumerate(primary_paths):
        target_d = round((i + 1) * ratio)
        while d_cursor < target_d:
            result.append(domain_slots[d_cursor])
            d_cursor += 1
        result.append(pf)
    result.extend(domain_slots[d_cursor:])  # any tail domain slots

    return result, report
```

### 2b. Modify `_document_batches` (`nanochat/dataloader.py`)

Change signature to accept optional `scheduled_paths`:

```python
def _document_batches(split, resume_state_dict, tokenizer_batch_size,
                      scheduled_paths=None):
```

Replace lines 35-37 (the `list_parquet_files()` + train/val split) with:

```python
if scheduled_paths is not None:
    parquet_paths = scheduled_paths   # caller handles val exclusion
else:
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

No other changes to `_document_batches` needed. The state_dict `pq_idx` now refers
to position in `scheduled_paths`, which is deterministic given the same inputs.

Update the public loader signatures to thread `scheduled_paths` through:

```python
def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split, ..., scheduled_paths=None
):
    ...
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size,
                                scheduled_paths=scheduled_paths)
```

### 2c. Wire up in `base_train.py`

Add CLI arg:
```python
parser.add_argument("--blend-m", type=int, default=10,
    help="oversampling multiplier for non-primary corpora (0 = disable blending)")
```

After `num_iterations` is computed (around line 338), insert:

```python
# --- Corpus blend ---
if args.blend_m > 0:
    from nanochat.dataset import get_corpuses, build_blend_schedule
    corpuses = get_corpuses()
    if len(corpuses) > 1:  # at least one non-primary corpus exists
        scheduled_paths, blend_report = build_blend_schedule(
            corpuses, m=args.blend_m, target_primary_tokens=target_tokens
        )
        # Extend training horizon to include domain oversampling
        domain_extra_tokens = sum(
            info["est_tokens"] * args.blend_m
            for name, info in blend_report.items() if name != "primary"
        )
        num_iterations = (target_tokens + domain_extra_tokens) // total_batch_size
        print0(f"Corpus blend enabled (m={args.blend_m}):")
        for name, info in blend_report.items():
            print0(f"  {name:20s}: {info['files']} files, "
                   f"~{info['est_tokens']/1e9:.2f}B est. tokens, "
                   f"repeats={info['repeats']}, "
                   f"eff. epochs≈{info.get('effective_epochs', 1):.1f}")
        print0(f"  Extended num_iterations: {num_iterations:,} "
               f"(+{domain_extra_tokens/1e9:.2f}B domain tokens)")
    else:
        scheduled_paths = None
else:
    scheduled_paths = None
```

Pass `scheduled_paths` to the dataloader:
```python
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict,
    scheduled_paths=scheduled_paths,   # NEW
)
```

Val loader stays unchanged (uses `split="val"`, last parquet file, no scheduled_paths).

---

## Phase 3: Feature 2 – Domain Q&A Generation Script

### 3a. New script: `scripts/generate_domain_qa.py`

Standalone script, run once before `chat_sft.py`.

**CLI args:**
```
--model-tag        str    base checkpoint to use (default: auto-detect largest)
--model-step       int    checkpoint step (default: latest)
--max-docs         int    max documents per corpus to process (default: 5000)
--qa-per-doc       int    Q&A pairs to generate per document chunk (default: 2)
--chunk-chars      int    document excerpt length in chars (default: 800)
--temperature      float  sampling temperature (default: 0.8)
--device-type      str    cuda|cpu|mps (default: autodetect)
--overwrite        flag   overwrite existing JSONL files
```

**Algorithm:**

1. Load base model via `load_model("base", device, phase="eval", ...)` (existing function).
2. Call `get_corpuses()` from `nanochat/dataset.py` to get non-primary corpora.
3. For each non-primary corpus:
   a. Iterate over parquet files via `parquets_iter_batched` (existing function).
   b. For each document, take a chunk of `chunk_chars` characters from the middle
      (avoid header/footer boilerplate often at start/end).
   c. Build few-shot prompt (hard-coded 3 examples):

```
Generate a question and detailed answer based on the following passage.

### Passage:
Photosynthesis is the process by which plants use sunlight, water, and CO₂
to produce sugar and oxygen. Chlorophyll in plant cells absorbs light energy.

### Question:
What role does chlorophyll play in photosynthesis?

### Answer:
Chlorophyll is the pigment in plant cells that absorbs light energy, which
is then used to convert water and CO₂ into sugar and oxygen.

---

### Passage:
{document_chunk}

### Question:
```

   d. Call `Engine.generate()` (existing inference class) with greedy or low-temperature
      sampling; parse the completion to extract Question and Answer strings.
   e. Filter: skip pairs where question < 10 chars, answer < 20 chars, or
      generation doesn't parse cleanly.
   f. Write to `{base_dir}/domain_qa/{corpus_name}.jsonl` in CustomJSON format:
      ```json
      [{"role":"user","content":"<question>"},{"role":"assistant","content":"<answer>"}]
      ```

4. Print summary: N pairs generated per corpus, N filtered, output path.

**Note on base model generation quality**: base models can follow few-shot format
reliably. Temperature ≈ 0.8 prevents repetition. The few-shot examples establish the
exact format expected from the completion.

### 3b. New task class: `tasks/domainqa.py`

```python
class DomainQA(Task):
    """Load domain Q&A pairs from JSONL files generated by generate_domain_qa.py."""
    # Almost identical to CustomJSON; reuse its __init__ and get_example logic.
    # Only difference: handles multiple JSONL files in a directory.
    eval_type = "generative"
    def __init__(self, jsonl_path, **kwargs): ...
    def num_examples(self): ...
    def get_example(self, index): ...
```

Actually: `DomainQA` can just be an alias for `CustomJSON` loaded with the right path.
A thin wrapper is sufficient:

```python
# tasks/domainqa.py
from tasks.customjson import CustomJSON

def get_domain_qa_tasks(base_dir, epochs=3):
    """Return list of Task objects for all domain Q&A corpora."""
    import os, glob
    qa_dir = os.path.join(base_dir, "domain_qa")
    tasks = []
    for jsonl_path in sorted(glob.glob(os.path.join(qa_dir, "*.jsonl"))):
        for _ in range(epochs):
            tasks.append(CustomJSON(filepath=jsonl_path))
    return tasks
```

### 3c. Enhance `chat_sft.py`

Add CLI arg:
```python
parser.add_argument("--domain-qa-epochs", type=int, default=3,
    help="epochs of domain Q&A pairs to include in SFT mixture (0 = skip)")
```

In the task mixture setup (after line 168), add:

```python
if args.domain_qa_epochs > 0:
    from tasks.domainqa import get_domain_qa_tasks
    domain_tasks = get_domain_qa_tasks(base_dir, epochs=args.domain_qa_epochs)
    if domain_tasks:
        train_tasks.extend(domain_tasks)
        total_qa = sum(len(t) for t in domain_tasks)
        print0(f"Added {len(domain_tasks)} domain Q&A task(s) × {args.domain_qa_epochs} epochs "
               f"= {total_qa:,} Q&A pairs")
    else:
        print0("No domain Q&A files found in {base_dir}/domain_qa/ — skipping")
```

---

## Files Modified / Created

| File | Action | Purpose |
|------|--------|---------|
| `nanochat/dataset.py` | Modify | Add `get_corpuses()`, `estimate_corpus_tokens()`, `build_blend_schedule()` |
| `nanochat/dataloader.py` | Modify | `_document_batches()` + public loaders accept `scheduled_paths` |
| `scripts/base_train.py` | Modify | `--blend-m` arg, blend schedule, extended `num_iterations` |
| `scripts/generate_domain_qa.py` | **New** | Offline Q&A generation using base model |
| `tasks/domainqa.py` | **New** | Thin helper returning `CustomJSON` tasks from `domain_qa/` dir |
| `scripts/chat_sft.py` | Modify | `--domain-qa-epochs` arg, auto-include domain Q&A tasks |

---

## End-to-End Usage (After Implementation)

```bash
# 1. Drop domain parquet files into base_data/ (any prefix other than "shard")
#    e.g. base_data/legal_00001.parquet, base_data/legal_00002.parquet

# 2. Pretrain with corpus blend (domain gets 10 full epochs, training extended automatically)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26 --blend-m=10

# 3. Generate domain Q&A pairs (run once, ~10-30 min on DGX Spark)
python -m scripts.generate_domain_qa --model-tag d26 --max-docs 5000 --qa-per-doc 2

# 4. SFT with domain Q&A included
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag d26 --domain-qa-epochs=3
```

---

## Verification

1. **Corpus detection**: `python -c "from nanochat.dataset import get_corpuses; print(get_corpuses())"` with a mix of `shard_*.parquet` and `legal_*.parquet` files.

2. **Schedule correctness**: Verify `build_blend_schedule()` returns correct k by checking `domain_slots` count = k × n_domain_files, and that primary+domain token ratio matches T_FW : m×S_domain.

3. **Dataloader smoke test**: Run base_train with `--num-iterations 5 --blend-m 10` on a tiny test corpus and verify mix in startup print.

4. **Q&A generation**: Run `generate_domain_qa.py` with `--max-docs 10` and inspect the output JSONL for correct format.

5. **SFT integration**: Run `chat_sft.py --domain-qa-epochs 1 --num-iterations 10` and verify the domain Q&A task appears in the mixture count.
