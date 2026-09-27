"""Add 'what differs' features (diff_features.py) to a stage-2 table.

Training table (has 'label'): noise-word scores are CROSS-FITTED - rows of fold k get scores
learned only from the other fold (same S1-hash folds as the stage-2 model), so the OOF
evaluation stays honest. The vocabulary learned on all training rows is saved for the test set.

usage: python stage2_diff.py <work_dir> <stage2.parquet> <out.parquet>
"""
import gc
import os
import sys
import time

import polars as pl

from diff_features import build_diff, learn_noise_vocab, load_vocab

REC = ["entity_id", "n_core", "a_norm", "a_nums"]
CH = 1_000_000


def attach_records(d, df):
    s1 = (pl.scan_parquet(os.path.join(d, "s1.parquet")).select(REC)
            .filter(pl.col("entity_id").is_in(df["s1"].unique().implode())).collect())
    o = (pl.concat([pl.scan_parquet(os.path.join(d, f"s{i}.parquet")).select(REC) for i in (2, 3)])
           .filter(pl.col("entity_id").is_in(df["o"].unique().implode())).collect())
    s1 = s1.rename({c: c + "_1" for c in REC[1:]})
    o = o.rename({c: c + "_o" for c in REC[1:]})
    return df.join(s1, left_on="s1", right_on="entity_id", how="left").join(o, left_on="o", right_on="entity_id", how="left")


def diff_in_chunks(x, vocab):
    parts = []
    for i in range(0, x.height, CH):
        parts.append(build_diff(x.slice(i, CH), vocab))
    return pl.concat(parts)


def main(d, table, out):
    t0 = time.time()
    df = pl.read_parquet(table)
    keys = df.select("s1", "o")
    x = attach_records(d, keys.with_row_index("_i"))
    x = x.with_columns(pl.col(c).fill_null("") for c in x.columns if c.endswith(("_1", "_o")))
    print("records attached", x.height, f"{time.time() - t0:.0f}s", flush=True)
    if "label" in df.columns:
        x = x.join(df.select("s1", "o", "label"), on=["s1", "o"])
        x = x.with_columns((pl.col("s1").hash(seed=11) % 2).cast(pl.Int8).alias("fold"))
        feats = []
        for k in (0, 1):
            vocab = learn_noise_vocab(x.filter(pl.col("fold") != k))
            part = x.filter(pl.col("fold") == k)
            feats.append(pl.concat([part.select("_i"), diff_in_chunks(part, vocab)], how="horizontal"))
            print(f"fold {k} diff features {time.time() - t0:.0f}s", flush=True)
            gc.collect()
        learn_noise_vocab(x)       # vocabulary from all training rows -> artifacts/noise_vocab.json
        f = pl.concat(feats).sort("_i").drop("_i")
    else:
        vocab = load_vocab()
        f = diff_in_chunks(x.sort("_i"), vocab)
    df = pl.concat([df, f], how="horizontal")
    df.write_parquet(out)
    print("written", out, df.height, df.width, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
