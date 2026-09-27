"""Blend the stage-2 probability with the cross-encoder score (LightGBM on 10 features).

usage:
  python ce_blend.py fit   <ce_dir> <train_gt.tsv>   # holdout: 2-fold OOF evaluation + fit artifacts/blend.txt
  python ce_blend.py apply <ce_dir> <out_scores.parquet>
<ce_dir> holds the ce_export.py files plus hold_ce.npy / test_ce.npy from the notebook.
Holdout result (training data, macro F0.5): stage 2 alone 0.9816, cross-encoder alone 0.9828, blend 0.9874.
"""
import os
import sys

import numpy as np
import polars as pl
import lightgbm as lgb

from common import ART, read_table
from evaluate import sweep


def feats(df):
    eps = 1e-6
    df = df.with_columns(lp=(pl.col('p').clip(eps, 1 - eps) / (1 - pl.col('p').clip(eps, 1 - eps))).log(),
                         ce=pl.col('ce').cast(pl.Float32))
    df = df.with_columns(
        ce_o_max=pl.col('ce').max().over('o'), ce_o_n=pl.len().over('o'),
        ce_s1_max=pl.col('ce').max().over('s1'), ce_s1_mean=pl.col('ce').mean().over('s1'),
        p_o_max=pl.col('p').max().over('o'))
    return df.with_columns(
        ce_o_second=pl.when(pl.col('ce_o_n') > 1).then(pl.col('ce').top_k(2).min().over('o')).otherwise(-20.0),
        ce_gap_o=pl.col('ce') - pl.col('ce_o_max'), p_gap_o=pl.col('p') - pl.col('p_o_max'),
        ce_gap_s1=pl.col('ce') - pl.col('ce_s1_max'))


F = ['lp', 'ce', 'ce_o_max', 'ce_o_n', 'ce_gap_o', 'p_gap_o', 'ce_o_second', 'ce_s1_max', 'ce_s1_mean', 'ce_gap_s1']
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, seed=1)
ROUNDS = 400


def load(ce_dir, name):
    x = pl.read_parquet(f'{ce_dir}/{name}_pairs_local.parquet').sort('row')
    return feats(x.with_columns(ce=pl.Series(np.load(f'{ce_dir}/{name}_ce.npy').astype(np.float32))))


if __name__ == '__main__':
    mode, ce_dir = sys.argv[1:3]
    if mode == 'fit':
        gt = read_table(sys.argv[3])
        x = load(ce_dir, 'hold').with_columns(f=pl.col('s1').hash(seed=77) % 2)
        H = gt.filter(pl.col('source1_entity_id').hash(seed=23) % 100 < 15)['source1_entity_id']
        oof = np.zeros(x.height)
        for k in (0, 1):
            tr = x.filter(pl.col('f') != k)
            m = lgb.train(P, lgb.Dataset(tr.select(F).to_numpy(), tr['label'].to_numpy()), ROUNDS)
            oof[(x['f'] == k).to_numpy()] = m.predict(x.filter(pl.col('f') == k).select(F).to_numpy())
        for name, sc in (('stage2 only', x.select('s1', 'o', 'p')), ('blend', x.select('s1', 'o', p=pl.Series(oof)))):
            r = sweep(sc, gt, H, grid=np.arange(0.5, 0.96, 0.025))
            print(name, 'best (threshold, macro F0.5):', max(r, key=lambda t: t[1]), flush=True)
        lgb.train(P, lgb.Dataset(x.select(F).to_numpy(), x['label'].to_numpy()), ROUNDS).save_model(os.path.join(ART, 'blend.txt'))
    else:
        x = load(ce_dir, 'test')
        m = lgb.Booster(model_file=os.path.join(ART, 'blend.txt'))
        x.select('s1', 'o', p=pl.Series(m.predict(x.select(F).to_numpy()))).write_parquet(sys.argv[3])
        print('scored', x.height)
