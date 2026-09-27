"""Pairwise features for (S1, S2/S3) candidate pairs.

All features are generic string / number comparisons, so they transfer to countries
that are absent from training (France). No country one-hot is used.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

REC_COLS = ["entity_id", "n_full", "n_core", "n_alt", "n_legal", "n_dom", "a_norm", "a_nums", "a_state"]


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _set_feats(a_list, b_list):
    """Jaccard / containment of whitespace token sets (python loop, but cheap)."""
    n = len(a_list)
    jac = np.zeros(n, np.float32)
    cont_a = np.zeros(n, np.float32)
    cont_b = np.zeros(n, np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            jac[i] = cont_a[i] = cont_b[i] = np.nan
            continue
        inter = len(sa & sb)
        jac[i] = inter / len(sa | sb)
        cont_a[i] = inter / len(sa)
        cont_b[i] = inter / len(sb)
    return jac, cont_a, cont_b


def _num_feats(a_list, b_list):
    n = len(a_list)
    first_eq = np.full(n, np.nan, np.float32)
    jac = np.full(n, np.nan, np.float32)
    cont = np.full(n, np.nan, np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        la, lb = a.split(), b.split()
        if not la or not lb:
            continue
        first_eq[i] = 1.0 if la[0] == lb[0] else (0.5 if la[0] in lb else 0.0)
        sa, sb = set(la), set(lb)
        inter = len(sa & sb)
        jac[i] = inter / len(sa | sb)
        cont[i] = inter / len(sa)
    return first_eq, jac, cont


def _fuzzy_cov(a_list, b_list, idf, default_idf):
    """Share of S1 tokens (plain and IDF-weighted) that have an exact or near-exact
    (similarity >= 85) counterpart in the other record: robust to typos and extra words."""
    n = len(a_list)
    cov = np.full(n, np.nan, np.float32)
    wcov = np.full(n, np.nan, np.float32)
    wexact = np.full(n, np.nan, np.float32)
    ratio = fuzz.ratio
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ta, tb = a.split(), b.split()
        if not ta or not tb:
            continue
        sb = set(tb)
        hit = wsum = whit = wex = 0.0
        for t in ta:
            w = idf.get(t, default_idf)
            wsum += w
            if t in sb:
                hit += 1
                whit += w
                wex += w
            elif len(t) >= 4 and any(ratio(t, u) >= 85 for u in tb):
                hit += 1
                whit += w
        cov[i] = hit / len(ta)
        wcov[i] = whit / wsum
        wexact[i] = wex / wsum
    return cov, wcov, wexact


def _legal_feat(a_list, b_list):
    out = np.full(len(a_list), np.nan, np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if a and b:
            sa, sb = set(a.split()), set(b.split())
            out[i] = 1.0 if sa == sb else (0.5 if sa & sb else 0.0)
    return out


def build(cand, s1, o, idf_name=None, idf_addr=None):
    """cand: DataFrame[s1, o, bscore_*, ranks]; s1/o: normalised record tables."""
    a = s1.select(REC_COLS).rename({c: c + "_1" for c in REC_COLS})
    b = o.select(REC_COLS).rename({c: c + "_o" for c in REC_COLS})
    df = cand.join(a, left_on="s1", right_on="entity_id_1").join(b, left_on="o", right_on="entity_id_o")
    c1, co = df["n_core_1"].to_list(), df["n_core_o"].to_list()
    alt = df["n_alt_o"].to_list()
    f1, fo = df["n_full_1"].to_list(), df["n_full_o"].to_list()
    ad1, ado = df["a_norm_1"].to_list(), df["a_norm_o"].to_list()
    cc1 = [x.replace(" ", "") for x in c1]
    cco = [x.replace(" ", "") for x in co]
    empty_o = np.array([not x for x in ado])
    feats = {
        "nm_ratio": _cp(c1, co, fuzz.ratio),
        "nm_tsort": _cp(c1, co, fuzz.token_sort_ratio),
        "nm_tset": _cp(c1, co, fuzz.token_set_ratio),
        "nm_partial": _cp(c1, co, fuzz.partial_ratio),
        "nm_wratio": _cp(c1, co, fuzz.WRatio),
        "nm_compact": _cp(cc1, cco, fuzz.ratio),
        "nm_compact_partial": _cp(cc1, cco, fuzz.partial_ratio),
        "nm_jw": _cp(c1, co, JaroWinkler.normalized_similarity),
        "nm_full_tsort": _cp(f1, fo, fuzz.token_sort_ratio),
        "nm_alt_tset": np.where(np.array([bool(x) for x in alt]), _cp(c1, alt, fuzz.token_set_ratio), np.nan).astype(np.float32),
        "ad_tset": _cp(ad1, ado, fuzz.token_set_ratio),
        "ad_tsort": _cp(ad1, ado, fuzz.token_sort_ratio),
        "ad_partial": _cp(ad1, ado, fuzz.partial_ratio),
    }
    feats["nm_jac"], feats["nm_cont1"], feats["nm_conto"] = _set_feats(c1, co)
    feats["ad_jac"], feats["ad_cont1"], feats["ad_conto"] = _set_feats(ad1, ado)
    feats["num_first"], feats["num_jac"], feats["num_cont1"] = _num_feats(df["a_nums_1"].to_list(), df["a_nums_o"].to_list())
    if idf_name is not None:
        dn, da = max(idf_name.values()), max(idf_addr.values())
        feats["nm_fcov"], feats["nm_wcov"], feats["nm_wexact"] = _fuzzy_cov(c1, co, idf_name, dn)
        feats["nm_fcov_o"], feats["nm_wcov_o"], _ = _fuzzy_cov(co, c1, idf_name, dn)
        feats["ad_fcov"], feats["ad_wcov"], feats["ad_wexact"] = _fuzzy_cov(ad1, ado, idf_addr, da)
        feats["ad_fcov_o"], feats["ad_wcov_o"], _ = _fuzzy_cov(ado, ad1, idf_addr, da)
        feats["ad_fcov"][empty_o] = np.nan
    feats["legal_eq"] = _legal_feat(df["n_legal_1"].to_list(), df["n_legal_o"].to_list())
    for k in ("ad_tset", "ad_tsort", "ad_partial"):
        feats[k][empty_o] = np.nan
    st1, sto = df["a_state_1"].to_list(), df["a_state_o"].to_list()
    feats["state_eq"] = np.array([np.nan if not x or not y else float(x == y) for x, y in zip(st1, sto)], np.float32)
    feats["len_core_1"] = np.array([len(x.split()) for x in c1], np.float32)
    feats["len_core_o"] = np.array([len(x.split()) for x in co], np.float32)
    feats["addr_empty_o"] = empty_o.astype(np.float32)
    feats["is_dom_o"] = df["n_dom_o"].cast(pl.Float32).to_numpy()
    feats["src3"] = np.array([x.startswith("S3") for x in df["o"].to_list()], np.float32)
    keep = [c for c in cand.columns]
    return df.select(keep).with_columns(pl.Series(k, v) for k, v in feats.items())
