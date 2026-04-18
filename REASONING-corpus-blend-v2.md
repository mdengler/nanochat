# Reasoning: Corpus Blend v2 Design Decisions

This document captures the reasoning chain that led to PLAN-corpus-blend-and-domain-QnA-SFT-v2.md,
so that discussion can continue across sessions.

---

## 1. Why decouple from `base_data_climbmix/`?

The original corpus blend feature (commit 5c9f787) discovered domain corpora by
looking for non-`shard_*` parquet files inside `DATA_DIR`. This broke during
rebase because upstream switched from FineWeb-EDU (`base_data/`) to ClimbMix
(`base_data_climbmix/`). The directory name is dataset-specific and will change
again. Domain data needs a stable home.

**Decision**: a separate `corpora/` directory under `base_dir`, with subdirectories
as corpus names. This is orthogonal to upstream dataset changes.

**Alternative considered**: a `--corpus-dir` flag pointing anywhere. Rejected as
over-flexible — a single conventional location is simpler and the flag can come later.

---

## 2. Why chunk text files? (The critical discovery)

We investigated the BOS-aligned bestfit dataloader (`nanochat/dataloader.py`
lines 126-155) and found that it **discards document tails** that don't fit:

```python
# When no doc fits remaining space, crop shortest to fill exactly
doc = doc_buffer.pop(shortest_idx)
row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], ...)
```

The cropped tail is never reused. The ~35% token loss cited in the docstring
assumes ClimbMix's web-page-length distribution. But a user's 10MB text file
(~2.5M tokens) would lose ~99.9% of its content — only the first ~2048 tokens
that happen to fill a row would be trained on.

**Decision**: `prepare_corpus.py` must chunk at ingestion time.

**Chunking strategy chosen**: split on paragraph boundaries (`\n\n`), greedily
merge consecutive paragraphs up to ~4000 chars (~1000 tokens). This is the
standard approach in FineWeb/Dolma-style data pipelines:
- Natural boundaries preserve semantic coherence
- ~1000 tokens per chunk is well under the 2048 context window
- The bestfit packer can then fit 1-2 chunks per row with minimal waste
- Short files (< 4000 chars) are left as single documents

**Alternatives considered**:
- Fixed character splits (e.g. every 4000 chars): loses mid-sentence, rejected
- Sentence-level splitting with nltk/spacy: adds heavy dependencies for marginal benefit
- One document per file with no chunking: would lose most content (see above)

---

## 3. Why skip-if-fresh rather than skip-if-exists?

The user requested: "skip if all parquet files are newer" as default.

**Decision**: compare newest source file mtime vs oldest output parquet mtime.
If all outputs are fresher than all inputs, skip. This handles the common case
(user adds new files to their source directory) without needing `--overwrite`.

**Rationale**: skip-if-exists would miss the case where a user adds new documents
to their source directory after initial preparation. Skip-if-fresh catches that
automatically. The standalone `prepare_corpus.py --overwrite` is the escape hatch
for forced regeneration.

---

## 4. Why `--corpus` on base_train.py (not just speedrun.sh)?

**Decision**: put `--corpus` on `base_train.py` as the primary interface, with
speedrun.sh as a thin passthrough via `CORPUS` env var.

**Reasoning**:
- Users who run `base_train.py` directly (the common case for custom training)
  get the feature without touching shell scripts
- Only rank 0 does file I/O, with a DDP barrier before `get_corpuses()`,
  so it's safe in multi-GPU training
- `prepare_one_corpus()` is extracted as an importable function so both the
  standalone script and base_train.py share the same code path

---

## 5. The k-formula fix (already committed, context for v2)

During the rebase review, we found that `build_blend_schedule()` had an inverted
ratio in the oversampling factor:

```python
# BEFORE (broken with large primary pools):
k = max(1, round(m * s_primary / max(target_primary_tokens, 1)))
# With ClimbMix-400B, d20 (893M target): k = round(10 * 400B / 893M) = 4,479

# AFTER (fixed):
k = max(1, round(m * target_primary_tokens / max(s_primary, 1)))
# k = round(10 * 893M / 400B) = 1 (sensible)
```

This was committed in `bfcc889`. The v2 plan builds on the corrected formula.

---

## 6. Corpus name derivation

**Decision**: sanitized basename of source directory, overridable with `--name`.

`/data/My Legal Docs/` -> `my_legal_docs`

Sanitization: lowercase, replace non-alphanumeric with `_`, collapse runs,
reject reserved names ("primary", "shard") by prefixing with "custom_".

---

## 7. Text extension detection

**Decision**: use a known list of extensions, not MIME type detection.

**Reasoning**: MIME detection (`python-magic`, `mimetypes`) is unreliable for
code files and markup, adds a dependency, and is slow on large directory trees.
A curated extension list covers the practical cases. Users with unusual file
types can rename to `.txt` or convert to parquet themselves.

---

## 8. Open questions for further discussion

- **Should chunking target size be configurable?** Current plan hardcodes ~4000 chars.
  A `--target-chunk-chars` flag on prepare_corpus.py would be easy to add.
  The default should match well with the 2048 context window.

- **Should we support `--corpus` pointing at a single file** (not just directories)?
  E.g. `--corpus=my_book.txt`. Low priority but trivial to add.

- **CSV handling**: currently treated as raw text (the whole CSV becomes one document
  or is chunked as text). Should we parse CSVs and treat each row as a document?
  Probably not — too many CSV formats, and users can preprocess to parquet.

- **Should the deprecation warning for legacy files in DATA_DIR be rate-limited?**
  Currently would print once per `get_corpuses()` call. In practice this is called
  once per training run so it's fine.
