"""Vectorised macro F0.5 (identical definition to metrics.py, but fast on millions of entities).

usage: python evaluate.py <scored.parquet with s1,o,p> <gt.parquet> [s1.parquet to restrict]
"""
import sys

import numpy as np
import polars as pl


def best_per_o(scored):
    return (scored.sort(["o", "p", "s1"], descending=[False, True, False])
                  .unique(subset="o", keep="first"))


def truth_pairs(gt):
    return (gt.filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("o"))
              .explode("o"))


def macro(pred_pairs, truth, all_s1):
    """pred_pairs, truth: DataFrame[s1, o]; all_s1: Series of S1 ids to average over."""
    base = pl.DataFrame({"s1": all_s1})
    nt = truth.group_by("s1").len("nt")
    npd = pred_pairs.group_by("s1").len("np")
    tp = pred_pairs.join(truth, on=["s1", "o"]).group_by("s1").len("tp")
    x = (base.join(nt, on="s1", how="left").join(npd, on="s1", how="left").join(tp, on="s1", how="left")
             .fill_null(0))
    p = x["tp"] / x["np"].clip(lower_bound=1)
    r = x["tp"] / x["nt"].clip(lower_bound=1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0)
    f = pl.select(pl.when(x["nt"] == 0).then((x["np"] == 0).cast(pl.Float64)).otherwise(f)).to_series()
    return float(f.mean()), x.with_columns(f.alias("f"))


def sweep(scored, gt, all_s1, grid=np.arange(0.30, 0.96, 0.025)):
    truth = truth_pairs(gt).filter(pl.col("s1").is_in(all_s1.implode()))
    # rows below the smallest threshold can never be predicted -> drop them first (exact, saves memory)
    best = best_per_o(scored.filter(pl.col("p") >= float(min(grid))))
    res = []
    for t in grid:
        m, _ = macro(best.filter(pl.col("p") >= t).select("s1", "o"), truth, all_s1)
        res.append((round(float(t), 3), round(m, 5)))
    return res


if __name__ == "__main__":
    scored = pl.read_parquet(sys.argv[1], columns=["s1", "o", "p"])
    from common import read_table
    gt = read_table(sys.argv[2])
    all_s1 = pl.read_parquet(sys.argv[3], columns=["entity_id"])["entity_id"] if len(sys.argv) > 3 else gt["source1_entity_id"]
    for t, m in sweep(scored, gt, all_s1):
        print(f"threshold {t:.3f}  macro F0.5 {m:.5f}")
