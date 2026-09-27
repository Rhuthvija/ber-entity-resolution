"""Per-entity decision: choose, for every S1 entity, the match set that maximises its
EXPECTED F0.5 given the model's probabilities (instead of one global threshold).

Each S2/S3 record is first given to its best S1 (one-owner rule). For an S1 entity with
candidate probabilities p1 >= p2 >= ... we consider "take the top k" for k = 0..n and pick
the k with the highest expected F0.5, approximating
    E[F(k)] ~ F0.5(tp = sum_{i<=k} p_i, predicted = k, true = sum_i p_i + m)
and for k = 0 the probability that the entity has no match at all, prod(1 - p_i) * s.
m (expected true matches the model never saw) and s (singleton prior scale) are tuned on
out-of-fold training predictions.
"""
import numpy as np
import polars as pl


def f05(tp, npred, ntrue):
    if npred == 0 or ntrue <= 0 or tp <= 0:
        return 0.0
    p, r = tp / npred, min(1.0, tp / ntrue)
    return 1.25 * p * r / (0.25 * p + r)


def decide(scored, m=0.1, s=1.0, floor=0.2):
    """scored: DataFrame[s1, o, p]. Returns DataFrame[s1, o] of accepted pairs."""
    best = (scored.sort(["o", "p", "s1"], descending=[False, True, False]).unique("o", keep="first")
                  .filter(pl.col("p") >= floor).sort(["s1", "p"], descending=[False, True]))
    g = best.group_by("s1", maintain_order=True).agg(pl.col("o"), pl.col("p"))
    out_s1, out_o = [], []
    for s1, os_, ps in g.iter_rows():
        ps = np.asarray(ps, dtype=np.float64)
        ntrue = ps.sum() + m
        best_k, best_v = 0, float(np.prod(1 - ps)) * s
        cum = 0.0
        for k in range(1, len(ps) + 1):
            cum += ps[k - 1]
            v = f05(cum, k, ntrue)
            if v > best_v:
                best_k, best_v = k, v
        for o in os_[:best_k]:
            out_s1.append(s1)
            out_o.append(o)
    return pl.DataFrame({"s1": out_s1, "o": out_o})
