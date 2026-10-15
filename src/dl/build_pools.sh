#!/bin/bash
# Пулы DL-сетей (4 сида FBShallow, stats=past) для протоколов оценки на train:
#   loseso — для каждой сессии S обучение на всех остальных 55;
#   chrono — только на сессиях раньше S (имя начинается с даты); при < MIN_CHRONO ранних сессий
#            пул не строится (модель откатывается на ЭЭГ без DL).
# Веса: data/models/dl/<proto>/<S>/fbp_s<k>.pt; numpy-артефакты:
#   data/features/subjpool_loseso/<S>/eeg_dl_<k>.npz, data/features/chrono/<S>/eeg_dl_<k>.npz
#   bash src/dl/build_pools.sh loseso|chrono
#   (переопределяются через окружение: ROOT — корень репозитория, PY, LOG, NGPU, NPROC, MIN_CHRONO;
#    входы — data/raw/Data и кэш src.dl.prep в $DL_SCRATCH/prep)
PROTO=$1
case "$PROTO" in loseso|chrono) ;; *) echo "usage: bash src/dl/build_pools.sh loseso|chrono" >&2; exit 2;; esac
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/home/vkborisov16/miniconda3/envs/kaggle/bin/python}
LOG=${LOG:-$ROOT/data/logs/dl_pools_$PROTO}
MIN_CHRONO=${MIN_CHRONO:-10}
mkdir -p $LOG
cd $ROOT
NAMES=$(ls data/raw/Data | sed 's/\.mat$//' | sort)
i=0; jobs=()
for S in $NAMES; do
  if [ "$PROTO" = chrono ]; then
    [ $i -lt $MIN_CHRONO ] && { i=$((i+1)); continue; }
    FOLD="chrono:$S"
  else
    FOLD="sess:$S"
  fi
  [ -f data/models/dl/$PROTO/$S/fbp_s3.pt ] || jobs+=("$((i % ${NGPU:-4})) $S $FOLD")
  i=$((i+1))
done
printf '%s\n' "${jobs[@]}" | xargs -P ${NPROC:-4} -L 1 bash -c '
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '"$PY"' -m src.dl.train --model fbshallow --win 2 \
    --norm ea --stats past --steps 3000 --seeds 0,1,2,3 --fold $2 --gpu $0 --tag fbp \
    --save-dir data/models/dl/'"$PROTO"'/$1 > '"$LOG"'/$1.log 2>&1'
OUT=data/features/$([ "$PROTO" = chrono ] && echo chrono || echo subjpool_loseso)
OMP_NUM_THREADS=1 $PY - <<PYEOF
from pathlib import Path
from src.dl.np_infer import export_to
for d in sorted(Path("data/models/dl/$PROTO").iterdir()):
    for k in range(4):
        p = d / f"fbp_s{k}.pt"
        if p.exists():
            export_to(p, Path("$OUT") / d.name / f"eeg_dl_{k}.npz")
print("export done")
PYEOF
