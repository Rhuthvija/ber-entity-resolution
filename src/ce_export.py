"""Export candidate pairs + romanised record text for the cross-encoder (run on a GPU, see colab/).

usage:
  python ce_export.py train <train_work_dir> <out_dir>   # needs stage2d.parquet (with label) + stage2d_oof2.parquet (stage-2 OOF scores)
  python ce_export.py test  <test_work_dir>  <out_dir>   # needs test_scores2.parquet (stage-2 scores)
Writes <name>_rec.parquet (idx, text) and <name>_pairs.parquet (row, a, b[, label]) for the notebook,
and <name>_pairs_local.parquet (with the entity ids) that ce_blend.py uses to map scores back.
Training split: 15% of S1 entities (hash) are held out; the cross-encoder is trained on pairs that share
neither an S1 nor an S2/S3 record with the holdout, and the holdout is used to fit/evaluate the blender.
"""
import multiprocessing as mp
import os
import sys

import polars as pl


def rom(chunk):
    from normalize import romanise
    return [romanise(n or '') + ' | ' + romanise(a or '') for n, a in chunk]


def _gen(d, ids):
    cols = ['entity_id', 'business_name', 'business_address']
    for s in ('s1', 's2', 's3'):
        r = pl.read_parquet(f'{d}/{s}.parquet', columns=cols).filter(pl.col('entity_id').is_in(ids.implode()))
        for i in range(0, r.height, 50000):
            c = r.slice(i, 50000)
            yield c['entity_id'].to_list(), list(zip(c['business_name'].to_list(), c['business_address'].to_list()))


def _job(x):
    return x[0], rom(x[1])


def texts(d, ids):
    parts = []
    with mp.get_context('spawn').Pool(2) as p:
        for eid, t in p.imap(_job, _gen(d, ids)):
            parts.append(pl.DataFrame({'entity_id': eid, 'text': t}))
    return pl.concat(parts).with_row_index('idx')


def ship(name, d, pairs, extra, out):
    ids = pl.concat([pairs['s1'], pairs['o']]).unique()
    rec = texts(d, ids)
    m = rec.select('entity_id', 'idx')
    pr = (pairs.join(m.rename({'entity_id': 's1', 'idx': 'a'}), on='s1')
               .join(m.rename({'entity_id': 'o', 'idx': 'b'}), on='o')).with_row_index('row')
    pr.write_parquet(f'{out}/{name}_pairs_local.parquet')
    rec.select('idx', 'text').write_parquet(f'{out}/{name}_rec.parquet', compression='zstd', compression_level=19)
    pr.select(['row', 'a', 'b'] + extra).write_parquet(f'{out}/{name}_pairs.parquet', compression='zstd', compression_level=19)
    print(name, 'pairs', pr.height, 'records', rec.height, flush=True)


if __name__ == '__main__':
    mode, d, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    if mode == 'train':
        tr = pl.read_parquet(f'{d}/stage2d.parquet', columns=['s1', 'o', 'label']).join(
            pl.read_parquet(f'{d}/stage2d_oof2.parquet', columns=['s1', 'o', 'p']), on=['s1', 'o'])
        tr = tr.with_columns(h=(pl.col('s1').hash(seed=23) % 100))
        ho = tr.filter(pl.col('h') < 15)['o'].unique()
        hold = tr.filter(pl.col('o').is_in(ho.implode()))
        train = tr.filter((pl.col('h') >= 15) & ~pl.col('o').is_in(ho.implode()))
        keep = train['s1'].unique().sample(fraction=1.0, seed=3)[:700000]
        train = train.filter(pl.col('s1').is_in(keep.implode()))
        ship('train', d, train.select('s1', 'o', 'label'), ['label'], out)
        ship('hold', d, hold.select('s1', 'o', 'label', 'p'), [], out)
    else:
        ship('test', d, pl.read_parquet(f'{d}/test_scores2.parquet', columns=['s1', 'o', 'p']), [], out)
