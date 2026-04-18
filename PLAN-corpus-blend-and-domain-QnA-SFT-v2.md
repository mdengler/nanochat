# Plan: Decoupled Corpus Storage + `--corpus` Ingestion Pipeline (v2)

## Problem

Domain parquets currently must go in `base_data_climbmix/` — an upstream-specific
directory name that will break on the next dataset switch (it already broke once
going from FineWeb-EDU's `base_data/` to ClimbMix's `base_data_climbmix/`).
Users also have to manually convert their text data to parquet format.

## Solution Overview

1. A new stable `~/out/nanochat/corpora/` directory with one subdirectory per corpus
2. A `prepare_corpus.py` script that ingests raw text/parquet directories with chunking
3. `--corpus` flag on `base_train.py` and `CORPUS` env var on `speedrun.sh`

## Target Directory Layout

```
~/out/nanochat/
├── base_data_climbmix/     # upstream primary shards (untouched)
│   ├── shard_00000.parquet
│   └── ...
└── corpora/                # user domain data (stable across upstream changes)
    ├── legal/
    │   ├── legal_00000.parquet
    │   └── legal_00001.parquet
    └── medical/
        └── medical_00000.parquet
```

---

## File Changes

### 1. `nanochat/dataset.py` — Decouple corpus discovery

**Add** `CORPORA_DIR` constant:
```python
CORPORA_DIR = os.path.join(base_dir, "corpora")
```

**Rewrite** `get_corpuses(data_dir=None, corpora_dir=None)`:

1. Scan `DATA_DIR` for primary `shard_*.parquet` files (exclude last as val shard)
2. Scan `CORPORA_DIR` subdirectories — each subdirectory name is a corpus name,
   all `*.parquet` files inside are that corpus's training files
3. **Backward compat**: if non-`shard_*` parquets exist in `DATA_DIR`, still
   include them but print a deprecation warning telling users to move to `corpora/{name}/`

```python
def get_corpuses(data_dir=None, corpora_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    corpora_dir = CORPORA_DIR if corpora_dir is None else corpora_dir

    all_paths = list_parquet_files(data_dir)
    train_paths = all_paths[:-1]  # exclude last (val shard)
    corpuses = {}

    # Primary shards
    primary = [p for p in train_paths if os.path.basename(p).startswith("shard_")]
    if primary:
        corpuses["primary"] = primary

    # Legacy: non-shard files in DATA_DIR
    legacy = [p for p in train_paths if not os.path.basename(p).startswith("shard_")]
    if legacy:
        print("DEPRECATION: non-primary parquets in DATA_DIR; move to corpora/{name}/")
        for path in legacy:
            prefix = os.path.basename(path).split('_')[0]
            corpuses.setdefault(prefix, []).append(path)

    # New: scan corpora/ subdirectories
    if os.path.isdir(corpora_dir):
        for entry in sorted(os.listdir(corpora_dir)):
            subdir = os.path.join(corpora_dir, entry)
            if not os.path.isdir(subdir):
                continue
            parquets = sorted(glob for glob in os.listdir(subdir)
                              if glob.endswith('.parquet'))
            parquets = [os.path.join(subdir, f) for f in parquets]
            if parquets:
                corpuses.setdefault(entry, []).extend(parquets)

    return corpuses
```

**No changes needed** to `nanochat/dataloader.py` — it receives `scheduled_paths`
(a flat list of file paths) and reads the `text` column from each. Works as-is.

**No changes needed** to `scripts/generate_domain_qa.py` — it calls `get_corpuses()`
and iterates non-primary corpora, which now come from `corpora/` automatically.

---

### 2. `scripts/prepare_corpus.py` — New file: directory-to-parquet ingestion

**Usage:**
```bash
python -m scripts.prepare_corpus /path/to/my/legal/docs
python -m scripts.prepare_corpus /path/to/my/legal/docs --name legal-v2
```

**Key design decisions:**

#### Text file extensions (known list, no MIME detection):
```python
TEXT_EXTENSIONS = {
    '.txt', '.md', '.rst', '.csv', '.tsv',
    '.json', '.jsonl', '.ndjson',
    '.html', '.htm', '.xml', '.sgml',
    '.log', '.cfg', '.ini', '.yaml', '.yml', '.toml',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
    '.go', '.rs', '.rb', '.pl', '.sh', '.bash', '.zsh',
    '.tex', '.bib', '.srt', '.vtt',
}
```

#### Chunking strategy (critical for training efficiency):

The BOS-aligned bestfit dataloader **discards** document tails that don't fit
in a row. A 10MB text file (~2.5M tokens) would lose 99.9% of its content.
Therefore `prepare_corpus.py` must chunk long documents at ingestion time.

Algorithm:
1. Split text on paragraph boundaries (double newline `\n\n`)
2. Greedily merge consecutive paragraphs up to `--target-chunk-chars` (default 4000,
   ~1000 tokens at ~4 chars/token, well under the 2048 context window)
3. Files already under the target size become a single document
4. Each chunk becomes one row in the output parquet `text` column

```python
TARGET_CHUNK_CHARS = 4000

def chunk_text(text, target_chars=TARGET_CHUNK_CHARS):
    """Split text into chunks at paragraph boundaries."""
    text = text.strip()
    if len(text) <= target_chars:
        return [text] if text else []
    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if current and current_len + para_len + 2 > target_chars:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + (2 if current_len > 0 else 0)
    if current:
        chunks.append('\n\n'.join(current))
    return chunks
```

#### Corpus name derivation:
- Default: sanitized basename of source directory
- `sanitize_name()`: lowercase, replace non-alphanumeric with `_`, collapse runs,
  prevent collision with reserved names ("primary", "shard")
- Override with `--name`

#### Skip-if-fresh logic (idempotency):
- Find the newest source file (text or parquet) by mtime
- Find the oldest output parquet in `corpora/{name}/` by mtime
- Skip if oldest output > newest source (all outputs are fresher than all inputs)
- Print message: "Corpus '{name}' is up to date, skipping (use --overwrite to force)"

#### Sharding:
- ~8000 rows per output shard (matches upstream density)
- Output filenames: `{corpus_name}_{00000}.parquet`, `{corpus_name}_{00001}.parquet`, ...

#### Parquet input handling:
- Parquet files in the source directory with a `text` column: read rows, apply
  the same chunking to any oversized texts, then re-shard into output
- Parquet files without a `text` column: skip with warning

---

### 3. `scripts/base_train.py` — Add `--corpus` CLI argument

**Argparse** (after `--blend-m`, around line 83):
```python
parser.add_argument("--corpus", type=str, action="append", default=[],
    help="path to a directory of text/parquet files to blend as a domain corpus (repeatable)")
```

**Preparation** (before the `get_corpuses()` call, around line 327):
```python
if args.corpus and master_process:
    from scripts.prepare_corpus import prepare_one_corpus
    for source_dir in args.corpus:
        prepare_one_corpus(source_dir)  # writes to corpora/{name}/, skips if fresh

if ddp:
    dist.barrier()  # ensure all ranks see prepared files
```

The actual `prepare_one_corpus()` function encapsulates the standalone script's
logic (find files, check freshness, chunk, write shards) so it can be called
both from CLI and programmatically.

---

### 4. `runs/speedrun.sh` — `CORPUS` env var

**Add** before the torchrun command (around line 108):
```bash
# Domain corpus directories (space-separated paths)
# Usage: CORPUS="/data/legal /data/medical" bash runs/speedrun.sh
CORPUS_ARGS=""
if [ -n "$CORPUS" ]; then
    for dir in $CORPUS; do
        CORPUS_ARGS="$CORPUS_ARGS --corpus=$dir"
    done
fi
```

**Append** `${CORPUS_ARGS}` to the base_train.py torchrun command.

---

### 5. `docs/custom-corpus-guide.md` — Update documentation

- Change paths from `base_data_climbmix/` to `corpora/{name}/`
- Add `prepare_corpus.py` usage examples
- Add `--corpus` flag examples
- Explain chunking behavior
- Remove manual parquet creation example (prepare_corpus.py replaces it)

---

## Backward Compatibility Matrix

| Scenario | Before | After |
|----------|--------|-------|
| Non-shard parquets in `base_data_climbmix/` | Works | Works + deprecation warning |
| Parquets in `corpora/legal/` | N/A | Works (new primary path) |
| No domain data anywhere | No blending | No blending |
| `--blend-m=0` | Disables blending | Still disables blending |
| `--corpus=/path/to/dir` | N/A | Auto-prepares + blends |

## Implementation Order

1. `nanochat/dataset.py` — add `CORPORA_DIR`, rewrite `get_corpuses()`
2. `scripts/prepare_corpus.py` — new file, standalone + importable
3. `scripts/base_train.py` — add `--corpus`, call prepare before `get_corpuses()`
4. `runs/speedrun.sh` — `CORPUS` env var passthrough
5. `docs/custom-corpus-guide.md` — update all paths and examples
