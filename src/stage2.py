"""Stage 2: collective (group-aware) re-scoring.

Stage 1 scores every (S1, S2/S3) pair in isolation. But a real business normally appears
several times in S2/S3, and those copies resemble each other, while decoy records (which
match no S1 entity) are mostly loners. Stage 2 adds features that look at the whole group:

  sibling features  - for pair (s1, o): how similar is o to the OTHER confident candidates of s1
                      (their stage-1 probability, name / address similarity, count of close siblings)
  twin features     - how many S2/S3 records in the split share o's exact name (+house number)
  group features    - per S1: number of confident candidates, sum / max of p;
                      per o: best competing S1 probability, margin, number of plausible S1s

A second LightGBM (2-fold out-of-fold on training, like stage 1) combines them with the
stage-1 probability. Everything uses only the records themselves - no labels at inference.

usage:
  python stage2.py features <work_dir> <scores.parquet> <out.parquet> [gt]   # build stage-2 table
  python stage2.py train    <train_stage2.parquet> <gt>                     # OOF eval + final model
  python stage2.py predict  <test_stage2.parquet> <test_source1.tsv> <out_dir>
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import ART, read_table

P_MIN = 0.01        # pairs below this stage-1 probability are dropped (they are never predicted)
G_MIN = 0.30        # a candidate counts as a "confident sibling" from this probability on
NCH = int(os.environ.get("S2_CHUNKS", 8))
PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("S2_LR", 0.03)),
              num_leaves=int(os.environ.get("S2_LEAVES", 255)), min_data_in_leaf=int(os.environ.get("S2_MINLEAF", 100)),
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count() or 2, verbose=-1, seed=17)
ROUNDS = int(os.environ.get("S2_ROUNDS", 700))
NON_FEAT = {"s1", "o", "label", "fold"}
TAG = os.environ.get("S2_TAG", "stage2")      # model name, so a 3rd round can be trained separately


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def records(d):
    cols = ["entity_id", "n_core", "a_norm", "a_nums"]
    o = pl.concat([pl.scan_parquet(os.path.join(d, f"s{i}.parquet")).select(cols) for i in (2, 3)])
    return o


def twin_counts(d):
    """How many S2/S3 records share the exact core name / core name + first house number."""
    o = records(d).with_columns(
        kn=pl.col("n_core"),
        kk=pl.col("n_core") + "|" + pl.col("a_nums").str.split(" ").list.first().fill_null(""),
    )
    o = o.with_columns(
        twin_name=pl.len().over("kn").cast(pl.Float32) - 1,
        twin_key=pl.len().over("kk").cast(pl.Float32) - 1,
    ).select("entity_id", "twin_name", "twin_key")
    return o.collect()


def build_features(d, scores_path, out_path, gt_path=None):
    t0 = time.time()
    sc = pl.scan_parquet(scores_path).select("s1", "o", "p").filter(pl.col("p") >= P_MIN).collect()
    # competition / group features
    sc = sc.with_columns(
        o_best=pl.col("p").max().over("o"),
        o_n_plaus=(pl.col("p") >= 0.2).sum().over("o").cast(pl.Float32),
        s1_n_conf=(pl.col("p") >= 0.5).sum().over("s1").cast(pl.Float32),
        s1_sum_p=pl.col("p").sum().over("s1"),
        s1_max_p=pl.col("p").max().over("s1"),
        o_rank=pl.col("p").rank("ordinal", descending=True).over("o").cast(pl.Float32),
    )
    second = sc.group_by("o").agg(pl.col("p").sort(descending=True).slice(1, 1).first().alias("o_second"))
    sc = sc.join(second, on="o", how="left").with_columns(pl.col("o_second").fill_null(0.0))
    sc = sc.with_columns(
        o_best_other=pl.when(pl.col("o_rank") == 1).then(pl.col("o_second")).otherwise(pl.col("o_best")),
    ).with_columns(o_margin=pl.col("p") - pl.col("o_best_other"))
    print("group feats", sc.height, f"{time.time() - t0:.0f}s", flush=True)
    tw = twin_counts(d)
    sc = sc.join(tw, left_on="o", right_on="entity_id", how="left")
    print("twin feats", f"{time.time() - t0:.0f}s", flush=True)
    del tw
    gc.collect()
    rec = records(d).select("entity_id", "n_core", "a_norm")
    # sibling features, chunked by S1
    parts = []
    sc = sc.with_columns((pl.col("s1").hash(seed=5) % NCH).alias("_c"))
    for k in range(NCH):
        c = sc.filter(pl.col("_c") == k).select("s1", "o", "p")
        ids = c["o"].unique().implode()
        r = rec.filter(pl.col("entity_id").is_in(ids)).collect(engine="streaming")
        c = c.join(r, left_on="o", right_on="entity_id")
        g = c.filter(pl.col("p") >= G_MIN).select(
            "s1", pl.col("o").alias("o2"), pl.col("p").alias("p2"),
            pl.col("n_core").alias("n2"), pl.col("a_norm").alias("a2"))
        x = c.join(g, on="s1").filter(pl.col("o") != pl.col("o2"))
        nm = _cp(x["n_core"].to_list(), x["n2"].to_list(), fuzz.token_sort_ratio)
        ad = _cp(x["a_norm"].to_list(), x["a2"].to_list(), fuzz.token_set_ratio)
        both = np.minimum(nm, ad)
        x = x.select("s1", "o", "p2").with_columns(
            pl.Series("sn", nm), pl.Series("sa", ad), pl.Series("sb", both))
        agg = x.group_by("s1", "o").agg(
            sib_n=pl.len().cast(pl.Float32),
            sib_nm_max=pl.col("sn").max(),
            sib_ad_max=pl.col("sa").max(),
            sib_both_max=pl.col("sb").max(),
            sib_nm_mean=pl.col("sn").mean(),
            sib_close=((pl.col("sn") >= 90) & (pl.col("sa") >= 80)).sum().cast(pl.Float32),
            sib_close_p=pl.when((pl.col("sn") >= 90) & (pl.col("sa") >= 80)).then(pl.col("p2")).otherwise(0.0).max(),
            sib_wsim=(pl.col("sb") * pl.col("p2")).sum() / pl.col("p2").sum(),
        )
        parts.append(c.select("s1", "o").join(agg, on=["s1", "o"], how="left"))
        print(f"sibling chunk {k} pairs {x.height} {time.time() - t0:.0f}s", flush=True)
        del x, g, c, r
        gc.collect()
    sib = pl.concat(parts)
    sc = sc.drop("_c").join(sib, on=["s1", "o"], how="left").with_columns(
        pl.col("sib_n").fill_null(0.0), pl.col("sib_close").fill_null(0.0))
    sc = sc.with_columns(
        logit_p=(pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log())
    if gt_path:
        gt = read_table(gt_path).filter(pl.col("matched_entity_ids") != "")
        gp = (gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("o"))
                .explode("o").with_columns(pl.lit(1, pl.Int8).alias("label")))
        sc = sc.join(gp, on=["s1", "o"], how="left").with_columns(pl.col("label").fill_null(0))
    sc.write_parquet(out_path)
    print("stage2 table", sc.height, sc.width, f"{time.time() - t0:.0f}s", flush=True)


def _fit(df, cols):
    ds = lgb.Dataset(df.select(cols).to_numpy(), label=df["label"].to_numpy(), feature_name=cols)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def train(table, gt_path, only_final=False):
    from evaluate import sweep
    t0 = time.time()
    df = pl.read_parquet(table).with_columns((pl.col("s1").hash(seed=11) % 2).cast(pl.Int8).alias("fold"))
    cols = [c for c in df.columns if c not in NON_FEAT]
    if only_final:
        oof_path = table.replace(".parquet", "_oof2.parquet")
        scored = pl.read_parquet(oof_path)
        gt = read_table(gt_path)
        res = sweep(scored, gt, gt["source1_entity_id"], grid=np.arange(0.30, 0.96, 0.025))
        del scored
        return _final(df, cols, max(res, key=lambda x: x[1]))
    oof = np.zeros(df.height, np.float32)
    fold = df["fold"].to_numpy()
    for k in (0, 1):
        tr = df.filter(pl.col("fold") == k)
        if tr.height > 6_000_000:
            tr = tr.sample(6_000_000, seed=1)
        m = _fit(tr, cols)
        idx = np.where(fold != k)[0]
        oof[idx] = m.predict(df[idx].select(cols).to_numpy(), num_threads=PARAMS["num_threads"])
        print(f"stage2 fold {k} {time.time() - t0:.0f}s", flush=True)
    scored = df.select("s1", "o").with_columns(pl.Series("p", oof))
    scored.write_parquet(table.replace(".parquet", "_oof2.parquet"))
    gt = read_table(gt_path)
    res = sweep(scored, gt, gt["source1_entity_id"], grid=np.arange(0.30, 0.96, 0.025))
    for t, m in res:
        print(f"stage2 threshold {t:.3f}  macro F0.5 {m:.5f}", flush=True)
    best = max(res, key=lambda x: x[1])
    print("STAGE2 BEST", best, flush=True)
    del scored, oof
    return _final(df, cols, best)


def _final(df, cols, best):
    gc.collect()
    samp = df if df.height <= 6_000_000 else df.sample(6_000_000, seed=2)
    del df
    gc.collect()
    final = _fit(samp, cols)
    final.save_model(os.path.join(ART, f"model_{TAG}.txt"))
    json.dump({"threshold": best[0], "oof_macro_f05": best[1], "features": cols},
              open(os.path.join(ART, f"model_{TAG}_meta.json"), "w"), indent=1)
    imp = sorted(zip(cols, final.feature_importance("gain")), key=lambda x: -x[1])
    print("importance", [(c, round(v)) for c, v in imp[:15]], flush=True)


def predict(table, out_scores, threshold=None):
    meta = json.load(open(os.path.join(ART, f"model_{TAG}_meta.json")))
    m = lgb.Booster(model_file=os.path.join(ART, f"model_{TAG}.txt"))
    df = pl.read_parquet(table)
    p = m.predict(df.select(meta["features"]).to_numpy(), num_threads=PARAMS["num_threads"]).astype(np.float32)
    df.select("s1", "o").with_columns(pl.Series("p", p)).write_parquet(out_scores)
    print("stage2 scored", df.height, "threshold", threshold or meta["threshold"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        build_features(*sys.argv[2:6])
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3], only_final=len(sys.argv) > 4 and sys.argv[4] == "final")
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
