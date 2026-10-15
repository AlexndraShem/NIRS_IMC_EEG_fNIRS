# DEPLOY — онлайн-декодер ЭЭГ+NIRS (решение хакатона)

Решение целиком лежит в `solution/` и не зависит от остального репозитория (`src/` судьёй не
используется). Контракт — `solution/classifier.py::OnlineClassifier` (ТЗ §3.2). Установка и запуск
рассчитаны на **замкнутое окружение без доступа в интернет**.

## 1. Окружение (офлайн)

- Python **3.11** (проверено на 3.11.11; ТЗ допускает 3.8–3.11), Linux x86_64 (glibc ≥ 2.17 —
  `python:3.11-slim` подходит).
- **Только CPU.** GPU/CUDA не нужны: нейросети ЭЭГ исполняются на numpy, torch не используется.
- Все зависимости лежат в репозитории готовыми колёсами — каталог `wheels/` (77 МБ). Установка
  к сети не обращается.

| пакет | версия | колесо в `wheels/` | размер |
|---|---|---|---|
| numpy | 1.26.4 | `numpy-1.26.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 18.3 МБ |
| scipy | 1.15.2 | `scipy-1.15.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 37.6 МБ |
| scikit-learn | 1.2.2 | `scikit_learn-1.2.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 9.6 МБ |
| joblib | 1.5.1 | `joblib-1.5.1-py3-none-any.whl` | 0.3 МБ |
| threadpoolctl | 3.6.0 | `threadpoolctl-3.6.0-py3-none-any.whl` | 0.02 МБ |
| pandas (нужен только официальному `score.py`) | 2.2.3 | `pandas-2.2.3-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 13.1 МБ |
| python-dateutil, pytz, tzdata, six (зависимости pandas) | 2.9.0.post0, 2025.2, 2025.2, 1.17.0 | `*-py2.py3-none-any.whl` | 1.1 МБ |

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links wheels -r requirements.txt      # сеть не нужна
python -c "import numpy, scipy, sklearn, pandas; print(numpy.__version__, scipy.__version__, sklearn.__version__, pandas.__version__)"
# ожидается: 1.26.4 1.15.2 1.2.2 2.2.3
```

`requirements.txt` (корень репозитория; копия — `solution/requirements.txt`) — строгие пины, без
диапазонов. Ставить именно их и в отдельный venv: scikit-learn 1.2.2 собран под numpy 1.x, с
numpy 2.x бинарно несовместим. `pip install --upgrade pip` не нужен (без сети он падает).

Проверено (2026-09-24): чистый venv на Python 3.11.11, `PIP_CONFIG_FILE=/dev/null pip install
--no-index --find-links wheels -r requirements.txt` → официальный `run.py` даёт побитово те же
`predictions.csv`, что среда разработки; официальный `score.py` работает.

## 2. Запуск (официальный CLI каркаса)

```bash
python solution/run.py   --input <session.mat> --output-dir <out> --artifacts solution/artifacts
python solution/score.py --session <session.mat> --predictions <out>/predictions.csv \
                         --timings <out>/timings.csv --output <out>/metrics.json
# каталог сессий (закрытый бакет) — цикл по *.mat:
for f in /data/*.mat; do s=$(basename "$f" .mat); python solution/run.py --input "$f" --output-dir out/$s; done
```

`run.py`, `io_utils.py`, `score.py` — замороженные файлы организаторов, судья подменяет их
эталонными; решение использует только `solution/classifier.py` (+ модули рядом и
`solution/artifacts/`). BLAS рекомендуется в 1 поток: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` (решение и само ставит 1 поток через threadpoolctl).

## 3. Асинхронный режим: где идут вычисления

Классификатор работает **асинхронно** — независимо от того, когда и как часто вызывается `predict`:

- `push(eeg_chunk, nirs_chunk)` принимает новые отсчёты, обновляет причинные фильтры (sosfilt с
  сохраняемым состоянием) и **на каждой точке сетки окон** (конец окна = 250 + 62·j отсчётов, т.е.
  каждые 250 мс, включая подготовку/паузы) считает вероятности ЭЭГ и NIRS и делает шаг слияния и
  сглаживания. Это основная вычислительная нагрузка: в среднем 6.5 мс, max 11 мс на точку сетки (раз в 250 мс).
- `predict()` только возвращает решение для последней точки сетки (≤ 250 мс назад), ничего не
  вычисляя и не меняя состояния: ≈ 3 мкс, max 0.08 мс.
- `fit_block(k, labels)` дообучает модели по меткам завершённого блока (ср. 0.64 с, max 0.89 с).

Официальный каркас хронометрирует только `predict` и `fit_block`; вычисления в `push` при этом
те же, что раньше шли в `predict` (одно решение раз в 250 мс), и укладываются в реальное время с
запасом ×30. Ответы на окнах судьи совпадают на 99.3 % при вызове `predict` на всех точках сетки и
подаче данных чанками по 20 отсчётов (остаток — только восстановление времени отсчётов NIRS, которое
судья не передаёт) — см. `docs/experiments/compliance_audit.md`.

## 4. Docker (сборка без сети)

`Dockerfile` (корень репозитория): `python:3.11-slim`, пользователь `jupyter` (uid 1000), зависимости — только из
`wheels/` (`pip install --no-index --find-links /tmp/wheels -r requirements.txt`), код в
`/home/jupyter/work/solution`. Базовый образ `python:3.11-slim` должен быть в локальном реестре.

```bash
docker build -t medic-bci-solution .
# или: DATA_DIR=/path/to/Data OUT_DIR=./out docker compose run --rm solve --input /data/<сессия>.mat --output-dir /out
docker run --rm -v /path/to/Data:/data:ro -v $PWD/out:/out medic-bci-solution \
  --input /data/<сессия>.mat --output-dir /out/<сессия>
```

(Сборка образа на машине разработки не проверялась — нет docker-демона; окружение проверено
чистым venv с установкой только из `wheels/`.)

## 5. Конфигурация модели и артефакты

`solution/artifacts/` (≈ 2.4 МБ) — всё обучено оффлайн на 56 открытых сессиях:

| файл | что это |
|---|---|
| `config.json` | какие модели подключать и с какими гиперпараметрами |
| `eeg_pool.npz` | общий пул-LR ЭЭГ base-модели (касательное пространство, 7 полос, 2 с) |
| `eeg_lrnb_pool.npz`, `eeg_lrsm4_pool.npz` | общие пулы ЭЭГ-экспертов: узкие полосы по биполярам/лапласианам; сенсомоторные каналы, 4 с |
| `eeg_subj/`, `eeg_lrnb_subj/`, `eeg_lrsm4_subj/` | пулы по прошлым сессиям участника (14 × `N??.npz`), выбираются по `meta["subject"]` |
| `nirs_subj/` | 12 sLDA NIRS по прошлым сессиям участника (14 × `N??.npz`) |
| `eeg_dl_0.npz` … `eeg_dl_3.npz` | 4 ЭЭГ-нейросети FBShallow (numpy-веса, по 141 КБ; окно 2 с, 125 Гц) |
| `fusion_meta.json` | стекинг-слияние: мета-LR по признакам τ ≤ 1 с (+ центрирование 60 с), обучены на оцениваемых окнах асинхронного прогона (EMA-признаки шагают по всем точкам сетки) |

Боевой конвейер: `artifact_guard:EEGArtifactGuard` (причинная охрана от ВЧ-артефактов ФЭС/ЭМГ) поверх
`eeg_dl:EEGWithDL` (`EEGLRExperts` base × nb × sm4 + 4 сети DL), `nirs_model:NirsCascadeEns`,
`fusion_stacked:StackedFusion`. Within-модели обучаются в `fit_block` на завершённых блоках; вес
пулов — по объёму размеченных данных сессии. Полное описание — `docs/experiments/final_pipeline.md`.

- Откат к синхронному режиму (решение в `predict`): `"async_ticks": false` в `config.json` — тогда
  нужен прежний `fusion_meta.json` синхронного режима (в репозиторий не входит).
- Откат охраны: `"eeg_model": "eeg_dl:EEGWithDL"` или `eeg.guard.enabled: false`.
- Откат слияния: `"fusion": "fusion:LogLinearFusion"` (или удалить `fusion_meta.json`).
- Без `subject` в meta — автоматически модели без пула участника.
- Если машина судьи медленнее: уменьшить `eeg.max_train` (400 → 300) и/или `nirs.fit_every: 2`.

## 6. Качество и время (официальный каркас, 56 train-сессий, 2026-09-24)

| протокол | 60F+30P | NIRS-бонус | штрафы | итог |
|---|---|---|---|---|
| **хронологический (официальная цифра)** | 59.19 | 11.62 | 0 | **70.81** |
| LOSessO | 62.53 | 12.15 | 0 | 74.68 |
| LOSubjO | 58.49 | 11.63 | 0 | 70.12 |

| вызов | среднее | max | лимит | штраф |
|---|---|---|---|---|
| `predict` | 0.003 мс | 0.08 мс | 0.20 с | 0 |
| `fit_block` (ЭЭГ + DL + NIRS) | 0.64 с | 0.89 с | 1.5 с | 0 |
| расчёт на точке сетки внутри `push` (судьёй не хронометрируется) | 6.5 мс | 11 мс | (0.25 с — реальное время) | — |

Хронологический прогон — 4 параллельных процесса на ненагруженной машине, BLAS в 1 поток. Под
сильной посторонней нагрузкой (load > 40) `fit_block` доходил до 2.8 с — при такой машине судьи
см. п. 5 «если машина медленнее».
