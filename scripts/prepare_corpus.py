"""
Ingest a directory of text or parquet files into a nanochat domain corpus.

Outputs go to {base_dir}/corpora/{name}/{name}_NNNNN.parquet, which is the
stable home for user domain data (decoupled from the upstream-specific
DATA_DIR like base_data_climbmix/).

Usage:
    python -m scripts.prepare_corpus /path/to/my/legal/docs
    python -m scripts.prepare_corpus /path/to/my/legal/docs --name legal-v2

Long documents are chunked at paragraph boundaries (~4000 chars / ~1000 tokens
per chunk) so the BOS-aligned bestfit dataloader doesn't discard their tails.
"""

import os
import re
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import CORPORA_DIR

# -----------------------------------------------------------------------------
# Defaults

# Approximate target chunk size in characters. ~4 chars per BPE token gives
# ~1000 tokens per chunk, comfortably under the 2048 default context window.
TARGET_CHUNK_CHARS = 4000

# Documents per output parquet shard (~matches upstream shard density).
SHARD_ROWS = 8000

# Filename extensions we will read as text. MIME detection is unreliable for
# code/markup and adds dependencies; users with unusual extensions can rename.
TEXT_EXTENSIONS = {
    '.txt', '.md', '.rst', '.csv', '.tsv',
    '.json', '.jsonl', '.ndjson',
    '.html', '.htm', '.xml', '.sgml',
    '.log', '.cfg', '.ini', '.yaml', '.yml', '.toml',
    '.py', '.js', '.ts', '.tsx', '.jsx',
    '.java', '.c', '.cc', '.cpp', '.h', '.hpp',
    '.go', '.rs', '.rb', '.pl', '.sh', '.bash', '.zsh',
    '.tex', '.bib', '.srt', '.vtt',
}

# Names that would collide with primary/shard handling in get_corpuses().
RESERVED_NAMES = {"primary", "shard"}

# -----------------------------------------------------------------------------
# Pure helpers

def sanitize_name(name):
    """Lowercase, collapse non-alphanumeric runs to underscore, avoid reserved names."""
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
    if not name:
        name = "custom"
    if name in RESERVED_NAMES or name.startswith("shard"):
        name = f"custom_{name}"
    return name


def chunk_text(text, target_chars=TARGET_CHUNK_CHARS):
    """Split text into chunks at paragraph boundaries (\\n\\n), greedy-merging
    consecutive paragraphs up to target_chars.

    Returns [] for empty input. A single paragraph larger than target_chars
    becomes one oversized chunk (we never split mid-paragraph).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        sep_len = 2 if current_len > 0 else 0
        if current and current_len + sep_len + para_len > target_chars:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += sep_len + para_len
    if current:
        chunks.append('\n\n'.join(current))
    return chunks


# -----------------------------------------------------------------------------
# Source-side iteration

def _iter_one_file(path, target_chunk_chars):
    """Yield document chunks from a single source file (text or parquet)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"  Skipping {path}: parquet read error: {e}")
            return
        if 'text' not in pf.schema_arrow.names:
            print(f"  Skipping {path}: parquet has no 'text' column")
            return
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=['text'])
            for t in rg.column('text').to_pylist():
                if t:
                    for chunk in chunk_text(t, target_chunk_chars):
                        yield chunk
    elif ext in TEXT_EXTENSIONS:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"  Skipping {path}: read error: {e}")
            return
        for chunk in chunk_text(text, target_chunk_chars):
            yield chunk
    # else: silently skip binary / unknown extensions


def iter_source_files(source):
    """Yield absolute paths of every file under `source` (recursive walk).

    Directory and filename ordering are forced alphabetical so the chunk order
    in the produced shards is reproducible across runs and filesystems.
    """
    if os.path.isfile(source):
        yield source
        return
    for root, dirs, files in os.walk(source):
        dirs.sort()
        for f in sorted(files):
            yield os.path.join(root, f)


def iter_documents(source, target_chunk_chars):
    """Yield document chunks from every text/parquet file under `source`."""
    for path in iter_source_files(source):
        yield from _iter_one_file(path, target_chunk_chars)


# -----------------------------------------------------------------------------
# Output-side writing

def _write_shard(buffer, output_dir, corpus_name, shard_idx):
    """Atomically write a parquet shard with a `text` column."""
    final_path = os.path.join(output_dir, f"{corpus_name}_{shard_idx:05d}.parquet")
    tmp_path = final_path + ".tmp"
    table = pa.table({"text": buffer})
    pq.write_table(table, tmp_path)
    os.rename(tmp_path, final_path)
    return final_path


def write_shards(documents_iter, output_dir, corpus_name, shard_rows=SHARD_ROWS):
    """Consume `documents_iter` and write parquet shards of `shard_rows` docs each."""
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    n_docs = 0
    buffer = []
    for doc in documents_iter:
        buffer.append(doc)
        n_docs += 1
        if len(buffer) >= shard_rows:
            _write_shard(buffer, output_dir, corpus_name, shard_idx)
            buffer = []
            shard_idx += 1
    if buffer:
        _write_shard(buffer, output_dir, corpus_name, shard_idx)
        shard_idx += 1
    return shard_idx, n_docs


# -----------------------------------------------------------------------------
# Skip-if-fresh

def is_fresh(source, output_dir):
    """Return True if all output parquets are newer than every source file.

    This is the default idempotency check: re-running prepare_corpus on the
    same source skips work, but adding a new file in the source dir invalidates
    freshness and triggers a rebuild.
    """
    if not os.path.isdir(output_dir):
        return False
    output_paths = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.parquet')
    ]
    if not output_paths:
        return False

    source_paths = list(iter_source_files(source))
    if not source_paths:
        # No source files: nothing to do, treat as fresh.
        return True

    newest_source = max(os.path.getmtime(p) for p in source_paths)
    oldest_output = min(os.path.getmtime(p) for p in output_paths)
    return oldest_output > newest_source


def _clear_existing_parquets(output_dir):
    """Remove any existing parquet files in output_dir prior to overwrite."""
    if not os.path.isdir(output_dir):
        return
    for f in os.listdir(output_dir):
        if f.endswith('.parquet') or f.endswith('.parquet.tmp'):
            os.remove(os.path.join(output_dir, f))


# -----------------------------------------------------------------------------
# Public entry point

def prepare_one_corpus(source, name=None, target_chunk_chars=TARGET_CHUNK_CHARS,
                       overwrite=False, shard_rows=SHARD_ROWS, corpora_dir=None):
    """Prepare one corpus from `source` (file or directory) into corpora/{name}/.

    Returns (name, output_dir).
    """
    source = os.path.expanduser(os.path.abspath(source))
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path does not exist: {source}")

    corpora_dir = CORPORA_DIR if corpora_dir is None else corpora_dir

    if name is None:
        basename = os.path.basename(source.rstrip(os.sep))
        if os.path.isfile(source):
            basename = os.path.splitext(basename)[0]
        name = sanitize_name(basename)
    else:
        name = sanitize_name(name)

    output_dir = os.path.join(corpora_dir, name)

    if not overwrite and is_fresh(source, output_dir):
        print(f"Corpus '{name}' is up to date at {output_dir}, skipping (use --overwrite to force)")
        return name, output_dir

    _clear_existing_parquets(output_dir)
    print(f"Preparing corpus '{name}' from {source} -> {output_dir}")
    n_shards, n_docs = write_shards(
        iter_documents(source, target_chunk_chars),
        output_dir, name, shard_rows,
    )
    if n_docs == 0:
        print(f"  Warning: no documents found in {source}")
    else:
        print(f"  Wrote {n_docs:,} document chunk(s) across {n_shards} shard(s) to {output_dir}")
    return name, output_dir


# -----------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a directory of text/parquet files into a nanochat domain corpus")
    parser.add_argument("source", type=str, help="Source directory (or single file) of text/parquet documents")
    parser.add_argument("--name", type=str, default=None,
        help="Corpus name (default: sanitized basename of source)")
    parser.add_argument("--target-chunk-chars", type=int, default=TARGET_CHUNK_CHARS,
        help=f"Approximate chunk size in characters (default: {TARGET_CHUNK_CHARS})")
    parser.add_argument("--shard-rows", type=int, default=SHARD_ROWS,
        help=f"Document chunks per output parquet shard (default: {SHARD_ROWS})")
    parser.add_argument("--overwrite", action="store_true",
        help="Force regeneration even if outputs are fresher than source")
    args = parser.parse_args()

    prepare_one_corpus(
        source=args.source,
        name=args.name,
        target_chunk_chars=args.target_chunk_chars,
        overwrite=args.overwrite,
        shard_rows=args.shard_rows,
    )
