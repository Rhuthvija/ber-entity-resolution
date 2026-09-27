"""Write the two submission files from the stage-2 model.

candidate_pairs.tsv  = exactly the pairs the final (stage-2) model runs inference on
                       (blocking candidates that survived the stage-1 learned filter, p1 >= P_MIN)
matching_results.tsv = each S2/S3 record -> its best S1 by stage-2 probability, kept if >= threshold

usage: python final_output.py <stage2_table.parquet> <stage2_scores.parquet> <test_source1.tsv> <out_dir> [threshold]
"""
import json
import os
import sys

import duckdb
import polars as pl

from common import ART
from predict import id_lists, write_tsv


def main(table, scores, s1_path, out_dir, thr=None):
    os.makedirs(out_dir, exist_ok=True)
    if thr is None:
        thr = json.load(open(os.path.join(ART, "model_stage2_meta.json")))["threshold"]
    thr = float(thr)
    tmp = os.path.dirname(os.path.abspath(scores))
    s1_ids = pl.read_csv(s1_path, separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])["entity_id"]
    con = duckdb.connect()
    con.execute(f"set memory_limit='{os.environ.get('DUCK_MEM', '4GB')}'; set temp_directory='{os.path.join(tmp, '_duck_tmp')}'; set preserve_insertion_order=false")
    write_tsv(id_lists(f"select s1, o from '{table}'", s1_ids, "candidate_entity_ids", con,
                       os.path.join(tmp, "_tmp_c.parquet")), os.path.join(out_dir, "candidate_pairs.tsv"))
    best = f"""select s1, o from (select s1, o, p, row_number() over (partition by o order by p desc, s1) rk
               from '{scores}') where rk = 1 and p >= {thr}"""
    res = id_lists(best, s1_ids, "matched_entity_ids", con, os.path.join(tmp, "_tmp_m.parquet"))
    write_tsv(res, os.path.join(out_dir, "matching_results.tsv"))
    print("threshold", thr, "S1", res.height, "with matches", (res["matched_entity_ids"] != "").sum())


if __name__ == "__main__":
    main(*sys.argv[1:6])
