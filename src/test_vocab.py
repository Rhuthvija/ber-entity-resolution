"""Extend the noise-word vocabulary to words never seen in training (e.g. French words),
WITHOUT labels, using the stage-1 probability as a soft label.

For a word w, the training vocabulary stores P(true match | w appears on only one side).
For words absent from training, we estimate the same quantity on the test candidate pairs as
the mean stage-1 probability of the pairs where w appears on only one side (stage 1 does not use
these word scores, so there is no circularity). On training data this soft estimate correlates
0.99 with the true label-based score; a monotone calibration (soft -> true), fitted on training
tokens, maps it onto the training scale.

usage:
  python test_vocab.py calibrate <train_work_dir>             -> artifacts/soft_calibration.json
  python test_vocab.py extend <test_work_dir> <out_vocab.json>  (uses stage2.parquet column p)
"""
import json
import os
import sys

import numpy as np
import polars as pl

from common import ART
from diff_features import _extra, clean_addr, load_vocab
from stage2_diff import attach_records

MIN_COUNT = 20
BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def token_stats(x, with_label):
    """soft (mean p) and, if available, true (mean label) per token, for the 4 vocab kinds."""
    st = {k: {} for k in ("n_core", "n_core_d", "a_norm", "a_norm_d")}
    cols = [x[c].to_list() for c in ("n_core_1", "n_core_o", "a_norm_1", "a_norm_o", "p")]
    lab = x["label"].to_list() if with_label else [0] * x.height
    for n1, no, a1, ao, p, l in zip(*cols, lab):
        a1, ao = clean_addr(a1), clean_addr(ao)
        for kind, toks in (("n_core", _extra(n1, no)), ("n_core_d", _extra(no, n1)),
                           ("a_norm", _extra(a1, ao) if ao else []), ("a_norm_d", _extra(ao, a1) if ao else [])):
            for t in toks:
                if kind.startswith("a_norm") and t.isdigit():
                    continue
                s = st[kind].setdefault(t, [0, 0.0, 0])
                s[0] += 1
                s[1] += p
                s[2] += l
    return st


def load_pairs(d, table, n=None, seed=5):
    cols = ["s1", "o", "p"] + (["label"] if "label" in pl.read_parquet(table, n_rows=1).columns else [])
    df = pl.read_parquet(table, columns=cols)
    if n and df.height > n:
        df = df.sample(n, seed=seed)
    x = attach_records(d, df.select("s1", "o").with_row_index("_i"))
    x = x.with_columns(pl.col(c).fill_null("") for c in ("n_core_1", "n_core_o", "a_norm_1", "a_norm_o"))
    return x.join(df, on=["s1", "o"])


def calibrate(train_dir):
    x = load_pairs(train_dir, os.path.join(train_dir, "stage2.parquet"), n=3_000_000)
    st = token_stats(x, with_label=True)
    soft, true = [], []
    for kind in st:
        for t, (n, sp, sl) in st[kind].items():
            if n >= MIN_COUNT:
                soft.append(sp / n)
                true.append(sl / n)
    soft, true = np.array(soft), np.array(true)
    centers, values = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (soft >= lo) & (soft < hi)
        if m.sum() >= 5:
            centers.append(float(soft[m].mean()))
            values.append(float(true[m].mean()))
    values = list(np.maximum.accumulate(values))          # monotone
    json.dump({"x": centers, "y": values, "corr": float(np.corrcoef(soft, true)[0, 1])},
              open(os.path.join(ART, "soft_calibration.json"), "w"), indent=1)
    print("calibration", list(zip(np.round(centers, 3), np.round(values, 3))), "corr", np.corrcoef(soft, true)[0, 1])


def extend(test_dir, out_path):
    cal = json.load(open(os.path.join(ART, "soft_calibration.json")))
    vocab = load_vocab()
    x = load_pairs(test_dir, os.path.join(test_dir, "stage2.parquet"))
    st = token_stats(x, with_label=False)
    added = {}
    for kind in st:
        vocab.setdefault(kind, {})
        n_add = 0
        for t, (n, sp, _) in st[kind].items():
            if n >= MIN_COUNT and t not in vocab[kind]:
                vocab[kind][t] = float(np.interp(sp / n, cal["x"], cal["y"]))
                n_add += 1
        added[kind] = n_add
    json.dump(vocab, open(out_path, "w"))
    print("added unseen words:", added)
    for w in ("gironde", "nord", "atlantique", "fils", "groupe", "associes", "france", "amicale"):
        print(" ", w, {k: round(vocab[k][w], 3) for k in vocab if w in vocab[k]})


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate(sys.argv[2])
    else:
        extend(sys.argv[2], sys.argv[3])
