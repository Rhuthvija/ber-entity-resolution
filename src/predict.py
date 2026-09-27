"""Score test candidate pairs and write the two submission files.

usage: python predict.py <work_test_dir> <test_source1.tsv> <output_dir>
writes <output_dir>/matching_results.tsv and <output_dir>/candidate_pairs.tsv
"""
import json
import os
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from common import ART


def id_lists(pairs_sql, s1_ids, col, con, tmp_path):
    """One row per S1 entity (in file order), comma-joined unique ids, '' when none.
    DuckDB sorts the (possibly ~40M) pairs with disk spilling; we then stream the sorted
    file and join ids per S1 entity, which keeps memory low."""
    con.execute(f"copy (select distinct s1, o from ({pairs_sql}) order by s1, o) to '{tmp_path}' (format parquet)")
    lists, cur, buf = {}, None, []
    for batch in pq.ParquetFile(tmp_path).iter_batches(batch_size=1_000_000, columns=["s1", "o"]):
        for a, b in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if a != cur:
                if cur is not None:
                    lists[cur] = ",".join(buf)
                cur, buf = a, []
            buf.append(b)
    if cur is not None:
        lists[cur] = ",".join(buf)
    os.remove(tmp_path)
    return pl.DataFrame({"source1_entity_id": s1_ids, col: [lists.get(s, "") for s in s1_ids.to_list()]})


def write_tsv(df, path):
    df.write_csv(path, separator="\t", quote_style="never", include_header=True)


def main(d, s1_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(ART, "model_meta.json")))
    model = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
    cols, thr = meta["features"], meta["threshold"]
    fdir = os.path.join(d, "feat")
    scores_path = os.path.join(d, "test_scores.parquet")
    scored = []
    for f in ([] if os.environ.get("REUSE_SCORES") and os.path.exists(scores_path) else sorted(os.listdir(fdir))):
        if not f.startswith("part-"):
            continue
        x = pl.read_parquet(os.path.join(fdir, f))
        p = model.predict(x.select(cols).to_numpy(), num_threads=os.cpu_count() or 2).astype(np.float32)
        # keep only pairs that could ever be predicted (p >= 0.05): exact for best-per-record + threshold
        scored.append(x.select("s1", "o").with_columns(pl.Series("p", p)).filter(pl.col("p") >= 0.05))
        del x
    if scored:
        pl.concat(scored).write_parquet(scores_path)
    del scored
    s1_ids = pl.read_csv(s1_path, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])["entity_id"]
    con = duckdb.connect()
    con.execute(f"set memory_limit='{os.environ.get('DUCK_MEM', '4GB')}'; set temp_directory='{os.path.join(d, '_duck_tmp')}'; set preserve_insertion_order=false")
    cand_sql = f"select s1, o from '{os.path.join(d, 'cand.parquet')}'"
    write_tsv(id_lists(cand_sql, s1_ids, "candidate_entity_ids", con, os.path.join(d, "_tmp_cand_sorted.parquet")), os.path.join(out_dir, "candidate_pairs.tsv"))
    # each S2/S3 record -> its single best S1 (ties: smaller S1 id), accepted if p >= threshold
    best_sql = f"""select s1, o from (
        select s1, o, p, row_number() over (partition by o order by p desc, s1) rk
        from '{scores_path}' where p >= {thr}) where rk = 1"""
    res = id_lists(best_sql, s1_ids, "matched_entity_ids", con, os.path.join(d, "_tmp_match_sorted.parquet"))
    write_tsv(res, os.path.join(out_dir, "matching_results.tsv"))
    n_match = (res["matched_entity_ids"] != "").sum()
    n_ids = con.execute(f"select count(*) from ({best_sql})").fetchone()[0]
    print(f"threshold {thr}  S1 rows {res.height}  with matches {n_match}  matched ids {n_ids}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
