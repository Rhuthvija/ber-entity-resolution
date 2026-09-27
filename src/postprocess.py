"""Turn pair probabilities into final match lists.

1. Each S2/S3 record belongs to at most one real business, so it is only allowed to
   match the S1 entity that gives it the highest probability (ties -> lower S1 id).
2. That best pair is accepted if its probability >= threshold.
"""
import polars as pl


def assign(scored, threshold):
    """scored: DataFrame[s1, o, p]. Returns {s1: set(o)} (S1 entities without matches omitted)."""
    best = (scored.sort(["o", "p", "s1"], descending=[False, True, False])
                  .unique(subset="o", keep="first", maintain_order=False)
                  .filter(pl.col("p") >= threshold))
    g = best.group_by("s1").agg(pl.col("o"))
    return {a: set(b) for a, b in g.iter_rows()}
