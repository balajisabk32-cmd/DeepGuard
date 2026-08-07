# DeepGuard — single image, two entrypoints (api + ui).
# Python 3.12 rather than 3.13: broadest wheel availability for mediapipe/opencv/librosa.
# Bump only after confirming every wheel resolves on the demo laptop.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# ffmpeg: normalization + PTS extraction (Appendix B). Both are hard requirements.
# libgl1 / libglib2.0-0: opencv-python runtime deps, absent from -slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits don't invalidate the pip cache.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Model weights are baked in, not downloaded at runtime.
# This is what makes `docker run --network none` pass (CP3 gate, §9 checklist).
COPY models/ ./models/

COPY config/ ./config/
COPY src/ ./src/
COPY eval/ ./eval/

EXPOSE 8000 8501

# Default entrypoint = API. The ui service overrides `command` in docker-compose.yml.
CMD ["uvicorn", "src.pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
