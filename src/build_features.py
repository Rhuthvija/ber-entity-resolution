"""Compute pair features for every candidate pair of a split, in chunks.

usage: python build_features.py <work_split_dir> [ground_truth.tsv_or_parquet]
writes <dir>/feat/part-XX.parquet (with a 'label' column when ground truth is given)
"""
import gc
import os
import sys
import time

import numpy as np
import polars as pl

from features import REC_COLS, build

NCHUNK = int(os.environ.get("NCHUNK", 32))


def load_gt(path):
    if path.endswith(".parquet"):
        gt = pl.read_parquet(path)
    else:
        from common import read_tsv
        gt = read_tsv(path)
    return (gt.filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("o"))
              .explode("o").with_columns(pl.lit(1, pl.Int8).alias("label")))


def idf_tables(s1, o):
    """Token IDF over all records of the split (S1 + S2 + S3); unsupervised, per split."""
    out = []
    for col in ("n_core", "a_norm"):
        docs = pl.concat([s1.select(col), o.select(col)])
        n = docs.select(pl.len()).collect().item()
        df = (docs.select(pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
                  .filter(pl.col("t") != "").group_by("t").len().collect())
        idf = np.log(1 + n / df["len"].to_numpy())
        out.append(dict(zip(df["t"].to_list(), idf.tolist())))
    return out


def main(d, gt_path=None):
    t0 = time.time()
    out = os.path.join(d, "feat")
    os.makedirs(out, exist_ok=True)
    # everything is scanned lazily per chunk to keep memory low (~2 GB peak)
    s1 = pl.scan_parquet(os.path.join(d, "s1.parquet")).select(REC_COLS)
    o = pl.concat([pl.scan_parquet(os.path.join(d, f"s{i}.parquet")).select(REC_COLS) for i in (2, 3)])
    if gt_path:
        gpq = os.path.join(out, "_gt_pairs.parquet")
        if not os.path.exists(gpq):
            load_gt(gt_path).write_parquet(gpq)
        gt = pl.scan_parquet(gpq)
    else:
        gt = None
    idf_name, idf_addr = idf_tables(s1, o)
    cand = pl.scan_parquet(os.path.join(d, "cand.parquet")).with_columns(
        (pl.col("s1").hash(seed=7) % NCHUNK).alias("_chunk"))
    for k in range(NCHUNK):
        dst = os.path.join(out, f"part-{k:02d}.parquet")
        if os.path.exists(dst):
            continue
        c = cand.filter(pl.col("_chunk") == k).drop("_chunk").collect(engine="streaming")
        ids1 = c["s1"].unique().implode()
        f = build(c, s1.filter(pl.col("entity_id").is_in(ids1)).collect(engine="streaming"),
                  o.filter(pl.col("entity_id").is_in(c["o"].unique().implode())).collect(engine="streaming"),
                  idf_name, idf_addr)
        if gt is not None:
            g = gt.filter(pl.col("s1").is_in(ids1)).collect(engine="streaming")
            f = f.join(g, on=["s1", "o"], how="left").with_columns(pl.col("label").fill_null(0))
        f.write_parquet(dst)
        print(f"chunk {k} rows {f.height} {time.time() - t0:.0f}s", flush=True)
        del f, c
        gc.collect()


if __name__ == "__main__":
    main(*sys.argv[1:3])
