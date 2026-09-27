"""Candidate generation (blocking): IDF-weighted inverted index with rare-key prefix filtering.

Every record becomes a bag of blocking keys:
  W:<token>   core-name tokens (legal forms / honorifics / stopwords removed), incl. DBA/"formerly" names
  C:<string>  whole core name with spaces removed (catches "lifeinvestments.com" vs "Life Investments")
  P:<prefix>  first 5 letters of long name tokens (robust to typos later in the word)
  A:<token>   address words (street names, localities, cities ...)
  N:<number>  address numbers with >= 2 digits (house / plot / flat numbers)
  G:<4-gram>  character 4-grams of the space-free core name: a typo only breaks the few
              fragments around it, so misspelt names still share most of their fragments
Keys are compared only inside the same country label (open set: any label works, incl. France).

The dataset reuses a limited vocabulary of names and streets, so single words are weak
keys. We therefore build COMPOSITE keys = pairs of a record's rarest single keys
(name x address, address x address, name x name). An S2/S3 record is usually a degraded
copy of its S1 record (parts of the address dropped, words shuffled or misspelt), so
S2/S3 records use only their few rarest keys (NK name, NA address) while S1 records use a
wider set (NK1 name, NA1 address). A true pair then shares at least one composite key
whenever the S2/S3 record kept a couple of its S1 record's rarer tokens.
Composite keys shared by too many records (CAP1 / CAPO) are dropped. Each pair is scored
by the IDF-style weight of the composite keys it shares; we keep, per S2/S3 record, its
top-K_O S1 records and, per S1 record, its top-K_1 S2/S3 records.

usage: python blocking.py <work_split_dir>   -> <dir>/cand.parquet
"""
import os
import shutil
import sys
import time

import duckdb

NK = int(os.environ.get("NK", 2))             # rarest name keys kept per S2/S3 record
NA = int(os.environ.get("NA", 3))             # rarest address keys kept per S2/S3 record
NK1 = int(os.environ.get("NK1", 3))           # rarest name keys kept per S1 record
NA1 = int(os.environ.get("NA1", 7))           # rarest address keys kept per S1 record
NAA1 = int(os.environ.get("NAA1", 5))
NG = int(os.environ.get("NG", 4))             # rarest name 4-grams kept per S2/S3 record (pairs of them = keys)
NG1 = int(os.environ.get("NG1", 8))           # rarest name 4-grams kept per S1 record         # S1 address x address pairs use only the top NAA1
SINGLE_DF = int(os.environ.get("SINGLE_DF", 3))  # a single key this rare is used on its own
CAP1 = int(os.environ.get("CAP1", 30))        # drop composite keys shared by more S1 records
CAPO = int(os.environ.get("CAPO", 150))       # ... or by more S2/S3 records
K_O = int(os.environ.get("K_O", 4))           # keep top-K S1 per S2/S3 record
K_1 = int(os.environ.get("K_1", 18))          # keep top-K S2/S3 per S1 record
MEM = os.environ.get("DUCK_MEM", "5GB")
THREADS = int(os.environ.get("DUCK_THREADS", os.cpu_count() or 2))


def keys_sql(src):
    return f"""
    with r as (select entity_id id, country, n_core, n_alt, a_norm, a_nums from {src}),
    ntok as (select id, country, unnest(string_split(n_core || ' ' || n_alt, ' ')) t from r),
    atok as (select id, country, unnest(string_split(a_norm, ' ')) t from r),
    nnum as (select id, country, unnest(string_split(a_nums, ' ')) t from r)
    select distinct id, country, key from (
        select id, country, 'W:' || t as key from ntok where length(t) >= 2
        union all select id, country, 'P:' || left(t, 5) from ntok where length(t) >= 7
        union all select id, country, 'C:' || replace(n_core, ' ', '') from r where length(replace(n_core, ' ', '')) >= 5
        union all select id, country, 'G:' || g from (
            select id, country, unnest(list_transform(range(1, length(c) - 2), i -> substring(c, i, 4))) g
            from (select id, country, replace(n_core, ' ', '') c from r) where length(c) >= 6)
        union all select id, country, 'A:' || t from atok
            where not regexp_matches(t, '^[0-9]+$') and (length(t) >= 3 or regexp_matches(t, '[0-9]'))
        union all select id, country, 'N:' || t from nnum where length(t) >= 2
    )
    """


def main(d):
    t0 = time.time()
    tmp = os.path.join(d, "_duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    dbf = os.path.join(d, "_block.duckdb")
    con = duckdb.connect(dbf)
    con.execute(f"set memory_limit='{MEM}'; set temp_directory='{tmp}'; set preserve_insertion_order=false; set threads={THREADS};")
    s1, s2, s3 = (f"'{os.path.join(d, f's{i}.parquet')}'" for i in (1, 2, 3))
    have = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    if "pairs_done" in have:
        print("resuming from saved pairs", flush=True)
        return select_candidates(con, d, t0, dbf, tmp)
    if "k1" not in have or "ko" not in have:
        # built in hash chunks: the DISTINCT over ~150M key rows does not fit in memory at once
        con.execute("create or replace table k1 (id varchar, country varchar, key varchar)")
        con.execute("create or replace table ko (id varchar, country varchar, key varchar)")
        for tbl, srcs, nch in (("k1", [s1], 3), ("ko", [s2, s3], 4)):
            for src in srcs:
                for h in range(nch):
                    part = f"(select * from {src} where hash(entity_id) % {nch} = {h})"
                    con.execute(f"insert into {tbl} {keys_sql(part)}")
    print("keys", round(time.time() - t0), "s", flush=True)
    con.execute("""
        create or replace table kw as
        with n as (select country, count(distinct id) n from ko group by all),
             dfo as (select country, key, count(*) dfo from ko group by all),
             df1 as (select country, key, count(*) df1 from k1 group by all)
        select country, key, ln(1 + n.n / dfo) w, dfo, df1
        from dfo join df1 using (country, key) join n using (country)
    """)
    # For every record keep its NK rarest name keys and NA rarest address keys (rarity = frequency
    # in S1), then form composite keys from all pairs of them (name x address, address x address,
    # name x name). Composite keys are very specific even when single words are common.
    for side, src, nk, na, naa, ng in (("1", "k1", NK1, NA1, NAA1, NG1), ("o", "ko", NK, NA, NA, NG)):
        con.execute(f"""
            create or replace table r{side} as
            select id, country, key, typ, df1, rn from (
                select *, row_number() over (partition by id, typ order by df1, dfo, key) rn from (
                    select {src}.id, country, key, df1, dfo,
                           case when key[1] in ('W', 'C', 'P') then 'n' when key[1] = 'G' then 'g' else 'a' end typ
                    from {src} join kw using (country, key)
                )
            )
            where rn <= case typ when 'n' then {nk} when 'g' then {ng} else {na} end
        """)
        con.execute(f"""
            create or replace table c{side} as
            select a.id, a.country, a.key || '|' || b.key ck
            from r{side} a join r{side} b on a.id = b.id and a.key < b.key
            where (a.typ = 'g' and b.typ = 'g')
               or (a.typ <> 'g' and b.typ <> 'g'
                   and (a.typ <> b.typ or a.typ = 'n' or (a.rn <= {naa} and b.rn <= {naa})))
            union all
            select id, country, key from r{side} where df1 <= {SINGLE_DF}
        """)
    print("composite keys", round(time.time() - t0), "s", flush=True)
    con.execute(f"""
        create or replace table cw as
        with d1 as (select country, ck, count(*) df1 from c1 group by all),
             do_ as (select country, ck, count(*) dfo from co group by all)
        select country, ck, df1, dfo, ln(1 + 1e6 / dfo) w from d1 join do_ using (country, ck)
        where df1 <= {CAP1} and dfo <= {CAPO}
    """)
    est = con.execute("select sum(df1*dfo)::bigint from cw").fetchone()[0]
    print("raw composite pairs", est, flush=True)
    # free space before the big join, then join in slices of S1 (each slice aggregates completely)
    for t in ("k1", "ko", "kw", "r1", "ro"):
        con.execute(f"drop table if exists {t}")
    con.execute("checkpoint")
    con.execute("create or replace table pairs (s1 varchar, o varchar, bscore double, nk bigint)")
    nsl = int(os.environ.get("PAIR_SLICES", 6))
    for h in range(nsl):
        con.execute(f"""
            insert into pairs
            select c1.id s1, co.id o, sum(cw.w) bscore, count(*) nk
            from cw join (select * from c1 where hash(id) % {nsl} = {h}) c1 using (country, ck)
                    join co using (country, ck)
            group by all
        """)
        print(f"  pair slice {h}", round(time.time() - t0), "s", flush=True)
    for t in ("c1", "co", "cw"):
        con.execute(f"drop table if exists {t}")
    con.execute("create table pairs_done as select 1 ok")
    con.execute("checkpoint")
    print("pairs", con.execute("select count(*) from pairs").fetchone()[0], round(time.time() - t0), "s", flush=True)
    return select_candidates(con, d, t0, dbf, tmp)


def select_candidates(con, d, t0, dbf, tmp):
    # top-K per record via hash aggregation (max_by with n), in slices to bound memory
    con.execute("create or replace table ostat as select o, max(bscore) mo, count(*) n_o from pairs group by o")
    con.execute("create or replace table sstat as select s1, max(bscore) m1, count(*) n_1 from pairs group by s1")
    con.execute("create or replace table sel0 (s1 varchar, o varchar)")
    nsl = int(os.environ.get("PAIR_SLICES", 6))
    for h in range(nsl):
        con.execute(f"""insert into sel0 select unnest(max_by(s1, bscore, {K_O})) s1, o
                        from pairs where hash(o) % {nsl} = {h} group by o""")
        con.execute(f"""insert into sel0 select s1, unnest(max_by(o, bscore, {K_1})) o
                        from pairs where hash(s1) % {nsl} = {h} group by s1""")
    con.execute("create or replace table sel as select distinct s1, o from sel0")
    con.execute("drop table sel0")
    print("selected", con.execute("select count(*) from sel").fetchone()[0], round(time.time() - t0), "s", flush=True)
    con.execute("""
        create or replace table cand as
        select p.s1, p.o, p.bscore, p.nk,
               row_number() over (partition by p.o order by p.bscore desc, p.s1) r_o,
               row_number() over (partition by p.s1 order by p.bscore desc, p.o) r_1,
               p.bscore / ostat.mo rel_o, p.bscore / sstat.m1 rel_1,
               ostat.n_o, sstat.n_1
        from sel join pairs p using (s1, o) join ostat using (o) join sstat using (s1)
    """)
    con.execute(f"copy cand to '{os.path.join(d, 'cand.parquet')}' (format parquet)")
    n = con.execute("select count(*) from cand").fetchone()[0]
    print("candidates", n, round(time.time() - t0), "s", flush=True)
    con.close()
    # the intermediate database is large (~8 GB); only cand.parquet is needed downstream
    os.remove(dbf)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
