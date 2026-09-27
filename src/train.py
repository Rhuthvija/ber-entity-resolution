"""Train the pair classifier (LightGBM) and choose the decision threshold for macro F0.5.

Honest validation: S1 entities are split into 2 folds by hash. Model A is trained on fold 0
and scores fold 1, model B the reverse, so every training pair gets an out-of-fold (OOF)
probability. The post-processing (each S2/S3 record goes to at most one S1 entity) and the
threshold are then tuned on those OOF scores with the exact challenge metric.
Finally one model is trained on all training data and saved for inference.

usage: python train.py <work_train_dir>
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import ART, read_table
from evaluate import sweep

NON_FEAT = {"s1", "o", "label", "fold"}
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", 6_000_000))
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count() or 2, verbose=-1, seed=7)
ROUNDS = int(os.environ.get("ROUNDS", 400))


def parts(d):
    p = os.path.join(d, "feat")
    return sorted(os.path.join(p, f) for f in os.listdir(p) if f.startswith("part-"))


def add_fold(f):
    return f.with_columns((pl.col("s1").hash(seed=11) % 2).cast(pl.Int8).alias("fold"))


def load_sample(d, frac):
    """Random sample of all feature parts (memory-friendly)."""
    out = [add_fold(pl.read_parquet(f)).sample(fraction=frac, seed=3) for f in parts(d)]
    return pl.concat(out)


def feat_cols(df):
    return [c for c in df.columns if c not in NON_FEAT]


def fit(df, cols):
    X = df.select(cols).to_numpy().astype(np.float32, copy=False)
    y = df["label"].to_numpy()
    ds = lgb.Dataset(X, label=y, feature_name=cols, free_raw_data=True)
    ds.construct()
    del X
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def main(d, gt_path):
    t0 = time.time()
    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in parts(d))
    df = load_sample(d, min(1.0, MAX_TRAIN_ROWS / total))
    cols = feat_cols(df)
    print("total", total, "sample", df.height, "features", len(cols), "pos", df["label"].sum(), flush=True)
    models = {k: fit(df.filter(pl.col("fold") == k), cols) for k in (0, 1)}
    print(f"fold models trained {time.time() - t0:.0f}s", flush=True)
    scored = []
    for f in parts(d):
        x = add_fold(pl.read_parquet(f))
        p = np.zeros(x.height, np.float32)
        fold = x["fold"].to_numpy()
        for k in (0, 1):
            idx = np.where(fold != k)[0]
            p[idx] = models[k].predict(x[idx].select(cols).to_numpy(), num_threads=PARAMS["num_threads"])
        scored.append(x.select("s1", "o", "label").with_columns(pl.Series("p", p)))
    scored = pl.concat(scored)
    scored.write_parquet(os.path.join(d, "oof.parquet"))
    print(f"oof scored {time.time() - t0:.0f}s", flush=True)
    del scored
    return df, cols


def evaluate_oof(d, gt_path):
    scored = pl.scan_parquet(os.path.join(d, "oof.parquet")).select("s1", "o", "p").filter(pl.col("p") >= 0.2).collect()
    gt = read_table(gt_path)
    res = sweep(scored, gt, gt["source1_entity_id"])
    for t, m in res:
        print(f"threshold {t:.3f}  macro F0.5 {m:.5f}", flush=True)
    return max(res, key=lambda x: x[1])


def final_fit(df, cols, best):
    t0 = time.time()
    final = fit(df, cols)
    os.makedirs(ART, exist_ok=True)
    final.save_model(os.path.join(ART, "model.txt"))
    json.dump({"threshold": best[0], "oof_macro_f05": best[1], "features": cols},
              open(os.path.join(ART, "model_meta.json"), "w"), indent=1)
    imp = sorted(zip(cols, final.feature_importance("gain")), key=lambda x: -x[1])
    print("importance", [(c, round(v)) for c, v in imp[:20]])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    d, g = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    if mode in ("all", "oof"):
        df, cols = main(d, g)
    best = evaluate_oof(d, g)
    print("BEST", best, flush=True)
    if mode == "oof":          # OOF scores + threshold only; the final model is fitted by a separate call
        sys.exit(0)
    if mode == "final":
        total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in parts(d))
        df = load_sample(d, min(1.0, MAX_TRAIN_ROWS / total)).drop("s1", "o", "fold")
        cols = feat_cols(df)
    final_fit(df, cols, best)
