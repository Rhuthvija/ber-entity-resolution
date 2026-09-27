#!/usr/bin/env bash
# End-to-end pipeline:
#   normalise -> blocking (composite rare keys) -> stage 1 (pair classifier, learned filter)
#   -> stage 2 (group-aware + "what differs" re-scoring) -> output files
#
#   bash run_pipeline.sh <dataset_dir> <output_dir> [work_dir]
#
#   <dataset_dir>  folder that contains train/ and test/ (the challenge's dataset/ folder)
#   <output_dir>   receives matching_results.tsv and candidate_pairs.tsv
#   [work_dir]     scratch space for intermediate files (default: ./work, needs ~30 GB)
#
# SKIP_TRAIN=1 reuses the shipped artifacts/ (transliteration dictionary, noise vocabulary,
# stage-1 and stage-2 models) and only runs the test split.
# THRESHOLD sets the final stage-2 decision threshold (default 0.775, see README).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$(cd "$1" && pwd)"
mkdir -p "$2" "${3:-$HERE/work}"
OUT="$(cd "$2" && pwd)"; WORK="$(cd "${3:-$HERE/work}" && pwd)"   # absolute: we cd into src/ below
cd "$HERE/src"
export BER_ROOT="$HERE"
# blocking budgets (see blocking.py) and stage-1 training sample
export NA=4 NG=3 NG1=6 MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-7000000} ROUNDS=500 DUCK_MEM=${DUCK_MEM:-4GB}
THRESHOLD="${THRESHOLD:-$([ -n "${CE_DIR:-}" ] && echo 0.75 || echo 0.775)}"
TR="$WORK/train"; TE="$WORK/test"; GT="$DATA/train/train_ground_truth.tsv"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[train 1/7] learn transliteration dictionary from training matches"
  python3 learn_translit.py "$DATA/train"
  echo "[train 2/7] normalise training records"
  python3 preprocess.py "$DATA/train" train "$TR"
  echo "[train 3/7] blocking"
  python3 blocking.py "$TR"
  python3 eval_blocking.py "$TR" "$GT"
  echo "[train 4/7] stage-1 features"
  python3 build_features.py "$TR" "$GT"
  echo "[train 5/7] stage-1 model: 2-fold out-of-fold scores, then final model"
  python3 train.py "$TR" "$GT" oof
  python3 train.py "$TR" "$GT" final
  echo "[train 6/7] stage-2 table: group features + cross-fitted 'what differs' features"
  python3 stage2.py features "$TR" "$TR/oof.parquet" "$TR/stage2.parquet" "$GT"
  python3 stage2_diff.py "$TR" "$TR/stage2.parquet" "$TR/stage2d.parquet"
  echo "[train 7/7] stage-2 model: 2-fold OOF threshold search, then final model"
  python3 stage2.py train "$TR/stage2d.parquet" "$GT" || python3 stage2.py train "$TR/stage2d.parquet" "$GT" final
  python3 test_vocab.py calibrate "$TR"     # soft-label -> true-score calibration for unseen words
  rm -rf "$TR/feat"
fi

echo "[test 1/5] normalise test records"
python3 preprocess.py "$DATA/test" test "$TE"
echo "[test 2/5] blocking + stage-1 features"
python3 blocking.py "$TE"
python3 build_features.py "$TE"
echo "[test 3/5] stage-1 scores"
python3 predict.py "$TE" "$DATA/test/test_source1.tsv" "$WORK/stage1_only"
echo "[test 4/5] stage-2 table + scores"
python3 stage2.py features "$TE" "$TE/test_scores.parquet" "$TE/stage2.parquet"
# words never seen in training (e.g. French) get noise scores estimated WITHOUT labels from stage-1 scores
python3 test_vocab.py extend "$TE" "$TE/noise_vocab_test.json"
NOISE_VOCAB="$TE/noise_vocab_test.json" python3 stage2_diff.py "$TE" "$TE/stage2.parquet" "$TE/stage2d.parquet"
python3 stage2.py predict "$TE/stage2d.parquet" "$TE/test_scores2.parquet"
SCORES="$TE/test_scores2.parquet"
# optional stage 3: cross-encoder re-scoring (the GPU step runs in colab/ber_cross_encoder.ipynb).
#   1) python3 ce_export.py train "$TR" "$WORK/ce"; python3 ce_export.py test "$TE" "$WORK/ce"
#   2) upload $WORK/ce/{train,hold,test}_{rec,pairs}.parquet to Drive, run the notebook, put hold_ce.npy + test_ce.npy in $WORK/ce
#   3) CE_DIR="$WORK/ce" bash run_pipeline.sh ...   (fits/uses artifacts/blend.txt)
if [ -n "${CE_DIR:-}" ] && [ -f "$CE_DIR/test_ce.npy" ]; then
  echo "[test 4b/5] blend stage-2 scores with the cross-encoder"
  [ "${SKIP_TRAIN:-0}" != "1" ] && python3 ce_blend.py fit "$CE_DIR" "$GT"
  python3 ce_blend.py apply "$CE_DIR" "$TE/test_blend.parquet"
  SCORES="$TE/test_blend.parquet"
fi
echo "[test 5/5] write submission files (threshold $THRESHOLD)"
python3 final_output.py "$TE/stage2d.parquet" "$SCORES" "$DATA/test/test_source1.tsv" "$OUT" "$THRESHOLD"
echo "done -> $OUT"
