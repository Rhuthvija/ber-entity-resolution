"""'What differs' features: the KIND of difference between two records, not just how much.

Noisy true copies and decoys (different businesses) differ in characteristic ways:
  * extra words in the S2/S3 name: generator noise words ("center", "services", "shri") are
    harmless, while real words ("group", "holdings", "care") signal a different business;
  * extra address words: an extra street type or state code means a different place;
  * house number: losing a leading/trailing digit is typical noise, a changed last digit
    typically means a neighbouring (different) business.
Token "noise scores" P(true match | token appears only in the S2/S3 record) are learned from
the TRAINING candidate pairs (learn_noise_vocab) and stored in artifacts/noise_vocab.json.
"""
import json
import os
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from common import ART

PRIOR = 0.02          # noise score for a token never seen in training
NUM_REL = ["equal", "missing", "drop_prefix", "drop_suffix", "add_prefix", "add_suffix",
           "sub_first", "sub_mid", "sub_last", "multi", "other"]
NUM_CODE = {k: i for i, k in enumerate(NUM_REL)}


def _extra(a, b):
    """tokens of b that have no exact or near (>=85) counterpart in a"""
    ta = a.split()
    sa = set(ta)
    out = []
    for t in set(b.split()):
        if t in sa:
            continue
        if len(t) >= 4 and any(fuzz.ratio(t, u) >= 85 for u in ta):
            continue
        out.append(t)
    return out


def num_rel(a, b):
    a = a.split()[0] if a else ""
    b = b.split()[0] if b else ""
    if not a or not b:
        return "missing"
    if a == b:
        return "equal"
    if a.endswith(b):
        return "drop_prefix"
    if a.startswith(b):
        return "drop_suffix"
    if b.endswith(a):
        return "add_prefix"
    if b.startswith(a):
        return "add_suffix"
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return "sub_first" if d[0] == 0 else ("sub_last" if d[0] == len(a) - 1 else "sub_mid")
        return "multi"
    return "other"


def _edit_code(u, v):
    """single-token change u -> v: code = op type (1..3) * 10 + position (0 first, 1 mid, 2 last);
    0 = word replaced entirely (more than 2 edits)."""
    from rapidfuzz.distance import Levenshtein
    ops = Levenshtein.editops(u, v)
    if not ops or len(ops) > 2:
        return 0.0
    op = ops[0]
    tag = {"replace": 1, "delete": 2, "insert": 3}[op.tag]
    pos = 0 if op.src_pos == 0 else (2 if op.src_pos >= len(u) - 1 else 1)
    return float(tag * 10 + pos + (100 if len(ops) == 2 else 0))


def learn_noise_vocab(pairs, min_count=20):
    """pairs: DataFrame[n_core_1, n_core_o, a_norm_1, a_norm_o, label] (training candidates).
    For every token: P(true match | token only in the S2/S3 record)  (keys n_core / a_norm)
    and P(true match | token only in the S1 record, i.e. dropped)    (keys n_core_d / a_norm_d)."""
    out = {}
    for col in ("n_core", "a_norm"):
        for suffix, flip in (("", False), ("_d", True)):
            tot, pos = {}, {}
            for a, b, l in zip(pairs[col + "_1"].to_list(), pairs[col + "_o"].to_list(), pairs["label"].to_list()):
                if not b:
                    continue
                for t in (_extra(b, a) if flip else _extra(a, b)):
                    if col == "a_norm" and t.isdigit():
                        continue
                    tot[t] = tot.get(t, 0) + 1
                    pos[t] = pos.get(t, 0) + l
            out[col + suffix] = {t: (pos[t] + PRIOR * 10) / (n + 10) for t, n in tot.items() if n >= min_count}
    json.dump(out, open(os.path.join(ART, "noise_vocab.json"), "w"))
    return out


def load_vocab(path=None):
    """training vocabulary, or an extended one (NOISE_VOCAB env var / explicit path)."""
    path = path or os.environ.get("NOISE_VOCAB") or os.path.join(ART, "noise_vocab.json")
    return json.load(open(path))


_NDEG = re.compile(r"\bndeg")


def clean_addr(s):
    """'N°' (romanised to 'ndeg') is a number sign, not an address word: 'ndeg4' -> '4'."""
    return _NDEG.sub(" ", s) if "ndeg" in s else s


def build_diff(df, vocab):
    """df: DataFrame[n_core_1, n_core_o, a_norm_1, a_norm_o, a_nums_1, a_nums_o] -> feature frame."""
    n = df.height
    f = {k: np.full(n, np.nan, np.float32) for k in (
        "nm_x_n", "nm_x_min", "nm_x_mean", "nm_d_n", "ad_x_n", "ad_x_min", "ad_x_mean", "ad_d_n",
        "num_rel", "num_x_n", "num_d_n", "nm_d_min", "ad_d_min", "nm_edit", "nm_first_eq", "nm_last_eq")}
    vn, va = vocab["n_core"], vocab["a_norm"]
    vnd, vad = vocab.get("n_core_d", {}), vocab.get("a_norm_d", {})
    c1, co = df["n_core_1"].to_list(), df["n_core_o"].to_list()
    a1 = [clean_addr(v) for v in df["a_norm_1"].to_list()]
    ao = [clean_addr(v) for v in df["a_norm_o"].to_list()]
    u1, uo = df["a_nums_1"].to_list(), df["a_nums_o"].to_list()
    for i in range(n):
        x = _extra(c1[i], co[i])
        f["nm_x_n"][i] = len(x)
        if x:
            s = [vn.get(t, PRIOR) for t in x]
            f["nm_x_min"][i], f["nm_x_mean"][i] = min(s), sum(s) / len(s)
        else:
            f["nm_x_min"][i] = f["nm_x_mean"][i] = 1.0
        dn = _extra(co[i], c1[i])
        f["nm_d_n"][i] = len(dn)
        f["nm_d_min"][i] = min([vnd.get(t, PRIOR) for t in dn]) if dn else 1.0
        t1, to = c1[i].split(), co[i].split()
        if t1 and to:
            f["nm_first_eq"][i] = float(t1[0] == to[0])
            f["nm_last_eq"][i] = float(t1[-1] == to[-1])
        if len(x) == 1 and len(dn) == 1:
            f["nm_edit"][i] = _edit_code(dn[0], x[0])
        if ao[i]:
            x = [t for t in _extra(a1[i], ao[i]) if not t.isdigit()]
            f["ad_x_n"][i] = len(x)
            if x:
                s = [va.get(t, PRIOR) for t in x]
                f["ad_x_min"][i], f["ad_x_mean"][i] = min(s), sum(s) / len(s)
            else:
                f["ad_x_min"][i] = f["ad_x_mean"][i] = 1.0
            da = [t for t in _extra(ao[i], a1[i]) if not t.isdigit()]
            f["ad_d_n"][i] = len(da)
            f["ad_d_min"][i] = min([vad.get(t, PRIOR) for t in da]) if da else 1.0
        f["num_rel"][i] = NUM_CODE[num_rel(u1[i], uo[i])]
        s1n, son = set(u1[i].split()), set(uo[i].split())
        if s1n and son:
            f["num_x_n"][i] = len(son - s1n)
            f["num_d_n"][i] = len(s1n - son)
    return pl.DataFrame(f)
