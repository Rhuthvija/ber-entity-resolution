"""Shared paths and I/O helpers."""
import os
import polars as pl

ROOT = os.environ.get("BER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")


def read_tsv(path):
    """Read a challenge TSV as all-string columns (no quoting, empty -> '')."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(pl.all().fill_null(""))


def read_table(path):
    """Read a TSV (challenge format) or a parquet file."""
    return pl.read_parquet(path) if path.endswith(".parquet") else read_tsv(path)
