# How to Add Your Own Corpus to Nanochat

## Quick Start

Point `--corpus` at any directory of text or parquet files and nanochat will
auto-ingest, chunk, and blend it into pretraining:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
  --depth=20 --blend-m=10 --corpus=/path/to/my/legal/docs
```

That's it. The corpus is named after the source directory (`legal_docs` here),
written to `~/.cache/nanochat/corpora/legal_docs/*.parquet`, and oversampled to
~10 effective epochs during training.

You can pass `--corpus` multiple times for multiple domains, e.g.
`--corpus=/data/legal --corpus=/data/medical`.

In `runs/speedrun.sh`, set the `CORPUS` env var to a space-separated list:

```bash
CORPUS="/data/legal /data/medical" bash runs/speedrun.sh
```

## Step 1: Prepare a Corpus (manual, if you'd rather)

The `--corpus` flag above calls `scripts/prepare_corpus.py` for you. You can
also run it directly to inspect or pre-stage corpora:

```bash
python -m scripts.prepare_corpus /path/to/my/legal/docs
python -m scripts.prepare_corpus /path/to/my/legal/docs --name legal-v2
```

It walks the source directory, reads any text-like files (`.txt`, `.md`,
`.py`, `.json`, `.html`, ...) and any parquet files with a `text` column,
**chunks long documents at paragraph boundaries** (~4000 chars / ~1000 tokens
each), and writes shards of ~8000 documents to:

```
~/.cache/nanochat/corpora/{name}/{name}_00000.parquet
~/.cache/nanochat/corpora/{name}/{name}_00001.parquet
...
```

Re-running on the same source is a no-op unless you add new files (skip-if-fresh).
Pass `--overwrite` to force regeneration.

### Why chunking matters

Nanochat's BOS-aligned bestfit dataloader discards document tails that don't
fit in a row. A 10MB text file (~2.5M tokens) would lose ~99.9% of its content
without chunking. Splitting at paragraph boundaries preserves semantic
coherence and lets the packer fit 1-2 chunks per row with minimal waste.

### Corpus directory layout

```
~/.cache/nanochat/
├── base_data_climbmix/        # upstream ClimbMix shards (untouched)
│   ├── shard_00000.parquet
│   └── ...
└── corpora/                   # your domain data — stable across upstream switches
    ├── legal/
    │   ├── legal_00000.parquet
    │   └── legal_00001.parquet
    └── medical/
        └── medical_00000.parquet
```

Each subdirectory of `corpora/` is one corpus, named after the subdirectory.
This is decoupled from `base_data_climbmix/` (which is upstream-specific and
will change again as nanochat moves to newer datasets).

## Step 2: Train with Blending

Once corpora are in `corpora/{name}/` (whether via `--corpus` or manual prep),
blending is automatic:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=20 --blend-m=10
```

`--blend-m=10` means each domain corpus will be oversampled so it gets ~10
effective epochs over the training run, while ClimbMix gets its
Chinchilla-optimal single pass. The training horizon is automatically extended
to accommodate the extra domain tokens.

## Step 3 (Optional): Domain Q&A for SFT

After pretraining, you can generate synthetic Q&A pairs from your domain corpus, then fine-tune:

```bash
# Generate Q&A
python -m scripts.generate_domain_qa --model-tag d20 --max-docs 5000

# SFT with domain Q&A included
torchrun --nproc_per_node=8 -m scripts.chat_sft --model-tag d20 --domain-qa-epochs=3
```

---

## Is There a Disadvantage to Training on Both?

**No — it's actively beneficial**, with one caveat.

**Pros of joint training:**
- Your model retains general language competence (grammar, reasoning, world knowledge) from ClimbMix
- Domain-specific knowledge is layered on top, not replacing general capabilities
- The blend schedule interleaves domain files across primary files, so the model sees both distributions throughout training (no catastrophic forgetting from sequential training)

**The caveat — capacity allocation:** A model has finite parameters. Tokens spent memorizing your domain are tokens that *could* have further reduced general loss. This is the "tax" of multi-domain training. But nanochat's approach (oversample domain corpus m times, extend training horizon proportionally) handles this well — you're not displacing ClimbMix tokens, you're adding domain tokens on top.

The dev/LOG.md entry from Feb 17 is instructive: attempts to mix other *general* web corpora (FinePDFs, DCLM) with FineWeb-EDU all underperformed pure FineWeb-EDU. But that's general-vs-general competition. Adding a *specific* domain corpus is a different proposition — you're teaching the model genuinely new information it can't get from the primary corpus.

---

## Scaling Analysis: Parameters, Data, and Domain Tokens

Nanochat uses `target_tokens = target_param_data_ratio * scaling_params` where the default ratio is 10.5 (between Chinchilla's ~20 and more aggressive modern ratios that undertrain per-param but are inference-optimal).

| | **GPT-2 scale** (d20) | **GPT-3 scale** (d96, hypothetical) | **Notes** |
|---|---|---|---|
| **Depth** | 20 | 96 | nanochat `--depth` |
| **model_dim** | 1280 | 6144 | depth * 64 |
| **Total params (P)** | ~124M | ~2.8B | incl. embeddings |
| **Scaling params** | ~85M | ~2.5B | transformer matrices + lm_head |
| **Chinchilla-optimal tokens (T)** | ~893M (ratio=10.5) | ~26.3B | 10.5 * scaling_params |
| **Batch size (B)** | ~524K (2^19) | ~2M (2^21) | B proportional to D^0.383 |
| **Training steps** | ~1,703 | ~12,500 | T / B |
| **Primary epochs (E)** | ~0.002 (400B pool) | ~0.066 | T / 400B |

Now, adding a domain corpus of size **S** tokens with `--blend-m=m`:

| Scenario | S (domain) | m | Domain tokens added | T' (total) | R_primary : R_domain | Extra steps |
|---|---|---|---|---|---|---|
| Small domain, GPT-2 scale | 10M | 10 | 100M | 993M | 8.9 : 1 | +191 (~11%) |
| Medium domain, GPT-2 scale | 100M | 10 | 1B | 1.89B | 0.9 : 1 | +1,907 (~112%) |
| Small domain, GPT-3 scale | 100M | 10 | 1B | 27.3B | 26.3 : 1 | +476 (~4%) |
| Medium domain, GPT-3 scale | 1B | 10 | 10B | 36.3B | 2.6 : 1 | +4,768 (~38%) |
| Large domain, GPT-3 scale | 10B | 10 | 100B | 126.3B | 0.26 : 1 | +47,684 (~381%) |

Where:
- **T' = T + S * m** (primary tokens + domain tokens * oversampling)
- **R_primary : R_domain = T : (S * m)**
- **E' = T' / B** (extended training steps)

### Discussion

**The key ratio is R_primary : R_domain.** When this drops below ~1:1, your domain corpus dominates training. This isn't necessarily bad if your goal is domain memorization, but the model's general capabilities may suffer.

**Practical recommendations:**

1. **For a small corpus (1-50M tokens):** Use `--blend-m=10` at d20. This is the sweet spot — your domain gets ~10 epochs of exposure, training extends modestly (~10-50%), and the model retains full general capability. This is what the existing nanochat blend feature was designed for.

2. **For a medium corpus (50M-500M tokens):** Consider reducing `--blend-m` to 3-5 to avoid the domain overwhelming the primary corpus. At d20, 500M * 10 = 5B domain tokens would be 5.6x the primary budget — far too heavy. `--blend-m=3` gives 1.5B, a more balanced ~1:1.7 ratio.

3. **For a large corpus (>1B tokens):** You likely want `--blend-m=1` (single pass) or even to treat it as additional primary data. At this scale, the corpus is large enough to learn from without oversampling.

4. **Scaling up depth helps.** A d20 model has ~85M scaling params and a ~893M token Chinchilla budget. If your domain corpus is 100M+ tokens, consider training at d26 or d32 — the larger model has more capacity to absorb both distributions without interference.

**One thing this approach does NOT do well:** true "memorization" in the sense of verbatim recall. Language models compress — they learn distributions, not exact strings. If you need the model to regurgitate specific facts reliably, the SFT stage (Step 3 with domain Q&A) is critical. Pretraining gives the model exposure; SFT with Q&A pairs teaches it to *retrieve and articulate* that knowledge on demand. The `--domain-qa-epochs=3` flag repeats each Q&A pair 3 times during SFT for exactly this reason.
