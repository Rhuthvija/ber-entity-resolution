"""Learn a native-script -> Latin token dictionary from TRAINING matches only.

For every training pair (S1 record, matched S2/S3 record) where the S2/S3 name or
address contains Indic script, each native token is aligned to the S1 token whose
spelling is closest to its rough romanisation (anyascii). Votes are counted and the
majority Latin spelling is kept. Output: artifacts/translit.json
"""
import json, os, re, sys, collections
import polars as pl
from anyascii import anyascii
from rapidfuzz import process, fuzz
from common import ART, read_tsv

ZW = re.compile(r"[​-‏﻿]")
NATIVE = re.compile(r"[ऀ-෿]")
TOK = re.compile(r"[^\s,.;:()\[\]{}\-/&'\"*#<>|!?]+", re.U)


def rough(t):
    return re.sub(r"[^a-z0-9]", "", anyascii(t).lower())


_PH = [("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
       ("sh", "s"), ("ch", "c"), ("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"), ("y", "i")]


def skel(x):
    """Consonant skeleton used to compare romanisations that differ in vowels."""
    for a, b in _PH:
        x = x.replace(a, b)
    x = x[:1] + re.sub(r"[aeiou]", "", x[1:])
    return re.sub(r"(.)\1+", r"\1", x)


def score(r, c, pos_match):
    s = max(fuzz.ratio(r, c), fuzz.ratio(skel(r), skel(c)))
    return s + (15 if pos_match else 0)


def main(data_dir):
    s1 = read_tsv(f"{data_dir}/train_source1.tsv").rename({"entity_id": "s1id"})
    o = pl.concat([read_tsv(f"{data_dir}/train_source2.tsv"), read_tsv(f"{data_dir}/train_source3.tsv")])
    o = o.filter(pl.col("business_name").str.contains(r"[ऀ-෿]") | pl.col("business_address").str.contains(r"[ऀ-෿]"))
    gt = read_tsv(f"{data_dir}/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "")
    gt = gt.select(pl.col("source1_entity_id").alias("s1id"), pl.col("matched_entity_ids").str.split(",").alias("entity_id")).explode("entity_id")
    p = gt.join(o, on="entity_id").join(s1, on="s1id", suffix="_1")
    print("native pairs", p.height, file=sys.stderr)
    votes = collections.defaultdict(collections.Counter)
    for col in ["business_name", "business_address"]:
        for a, b in zip(p[col].to_list(), p[col + "_1"].to_list()):
            if not a or not NATIVE.search(a):
                continue
            cands = [t.lower() for t in TOK.findall(b)]
            if not cands:
                continue
            toks = TOK.findall(ZW.sub("", a))
            same_len = len(toks) == len(cands)
            for i, t in enumerate(toks):
                if not NATIVE.search(t):
                    continue
                r = rough(t)
                if not r:
                    continue
                best, bs = None, 0
                for j, c in enumerate(cands):
                    if c.isdigit():
                        continue
                    sc = score(r, c, same_len and i == j)
                    if sc > bs:
                        best, bs = c, sc
                if best and bs >= 55:
                    votes[t][best] += 1
    out = {}
    for t, c in votes.items():
        (w, n), tot = c.most_common(1)[0], sum(c.values())
        if n >= 2 and n / tot >= 0.5:
            out[t] = w
    os.makedirs(ART, exist_ok=True)
    json.dump(out, open(os.path.join(ART, "translit.json"), "w"), ensure_ascii=False)
    print("learned", len(out), "of", len(votes), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
