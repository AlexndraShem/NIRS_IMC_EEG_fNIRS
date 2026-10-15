# Образ сдачи: Python 3.11 + пользователь jupyter (uid 1000) — требование DataSphere.
# Сборка БЕЗ доступа в интернет: зависимости ставятся только из колёс репозитория (wheels/).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN useradd -m -u 1000 -s /bin/bash jupyter

WORKDIR /home/jupyter/work
COPY wheels/ /tmp/wheels/
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links /tmp/wheels -r /tmp/requirements.txt && rm -rf /tmp/wheels

COPY --chown=jupyter:jupyter solution/ /home/jupyter/work/solution/

USER jupyter
# Аргументы run.py передаются при запуске (см. DEPLOY.md, docker-compose.yml)
ENTRYPOINT ["python", "solution/run.py"]
CMD ["--help"]
