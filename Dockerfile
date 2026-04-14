# Browser-only users: build and run — no host conda required.
#   docker build -t interoptimus-web .
#   docker run --rm -p 8765:8765 interoptimus-web
# Then open http://127.0.0.1:8765
#
# MatRIS / custom MLIP wheels are not included; extend this image or mount checkpoints.

FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INTEROPTIMUS_NO_AUTO_VENV=1

COPY pyproject.toml setup.py README.md LICENSE ./
COPY InterOptimus ./InterOptimus

RUN pip install --upgrade pip wheel \
    && pip install -e ".[web]"

EXPOSE 8765
CMD ["python", "-m", "InterOptimus.web.app", "--host", "0.0.0.0", "--port", "8765"]
