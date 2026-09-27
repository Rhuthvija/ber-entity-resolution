"""Normalise every record of one split into a parquet table used by all later stages.

usage: python preprocess.py <dataset_split_dir> <prefix: train|test> <out_dir>
"""
import os
import sys
from multiprocessing import get_context

import polars as pl

from common import read_tsv
from normalize import norm_name, norm_addr

COLS = ["n_full", "n_core", "n_alt", "n_legal", "n_dom", "a_norm", "a_nums", "a_state"]
STEP = 40_000


def _work(rows):
    out = {c: [] for c in COLS}
    for name, addr, ctry in rows:
        d = norm_name(name)
        d.update(norm_addr(addr, ctry))
        for c in COLS:
            out[c].append(d[c])
    return out


def _rows(df):
    return list(zip(df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()))


def main(split_dir, prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nproc = max(1, os.cpu_count() or 1)
    for s in (1, 2, 3):
        src = os.path.join(split_dir, f"{prefix}_source{s}.tsv")
        dst = os.path.join(out_dir, f"s{s}.parquet")
        if os.path.exists(dst):
            continue
        df = read_tsv(src)
        n = df.height
        starts = list(range(0, n, STEP))
        tmp = os.path.join(out_dir, f"_s{s}_parts")
        os.makedirs(tmp, exist_ok=True)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        chunks = (_rows(df.slice(i, STEP)) for i in starts)
        # 'spawn' (not fork): forking a process that already runs polars threads can deadlock
        with get_context("spawn").Pool(nproc) as p:
            for k, out in enumerate(p.imap(_work, chunks)):
                part = df.slice(starts[k], STEP).with_columns(pl.Series(c, v) for c, v in out.items())
                part = part.with_columns(pl.col("n_dom").cast(pl.Int8))
                part.write_parquet(os.path.join(tmp, f"{k:05d}.parquet"))
        del df
        pl.scan_parquet(os.path.join(tmp, "*.parquet")).sink_parquet(dst)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
        print(src, n, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
