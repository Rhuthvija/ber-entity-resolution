"""Blocking recall on the training split: share of true pairs kept in cand.parquet."""
import sys
import duckdb
d, gt_path = sys.argv[1], sys.argv[2]
gt_src = f"read_csv('{gt_path}', delim='\t', quote='', header=true, all_varchar=true)" if gt_path.endswith(".tsv") else f"'{gt_path}'"
c = duckdb.connect()
c.execute(f"""create view gt as select source1_entity_id s1, unnest(string_split(matched_entity_ids, ',')) o
              from {gt_src} where coalesce(matched_entity_ids, '') <> ''""")
c.execute(f"create view cand as select * from '{d}/cand.parquet'")
c.execute(f"create view s1 as select entity_id, country from '{d}/s1.parquet'")
print(c.sql("""select s1.country, count(*) n_true, count(cand.o) found, round(count(cand.o)/count(*),4) recall
               from gt join s1 on gt.s1=s1.entity_id left join cand using (s1, o) group by all order by 1"""))
print(c.sql("select count(*) cands, count(distinct s1) s1_with_cands, round(count(*)/(select count(*) from s1),2) per_s1 from cand"))
print(c.sql("""select left(o,2) src, count(*) n_true, round(count(cand.o)/count(*),4) recall from gt left join cand using (s1,o) group by 1"""))
