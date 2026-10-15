#!/bin/bash
# Прогон одной конфигурации по 14 фолдам LOSubjO на 4 GPU (4 процесса одновременно).
#   bash src/dl/run_loso.sh <tag> <аргументы src.dl.train...>
TAG=$1; shift
PY=${PY:-/home/vkborisov16/miniconda3/envs/kaggle/bin/python}
LOG=${LOG:-/place/vartmp/claude-740548/-home-vkborisov16-projects-hackatons-medic/62fd6f60-50fc-45a1-9add-e0b5b7dac57f/scratchpad/dl/logs}
mkdir -p $LOG
FOLDS=${FOLDS:-"N01 N02 N03 N06 N08 N09 N10 N11 N13 N14 N15 N16 N17 N18"}
i=0
for f in $FOLDS; do echo "$((i % 4)) $f"; i=$((i+1)); done | \
  xargs -P ${NPROC:-4} -L 1 bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '"$PY"' -m src.dl.train --fold $1 --gpu $0 --tag '"$TAG"' '"$*"' > '"$LOG/$TAG"'_$1.log 2>&1'
grep -h FINAL $LOG/${TAG}_*.log
