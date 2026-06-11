# Use CPU base image for faster build. Switch to NVIDIA CUDA runtime for GPU-specific customizations if needed.
FROM ultralytics/ultralytics:8.4.58-python-export

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV TRAINER_PACKAGE_DIR=/opt/recipe
WORKDIR $TRAINER_PACKAGE_DIR

# Copy dependency files first for better layer caching.
COPY pyproject.toml README.md .python-version ./
RUN uv sync --no-cache

# Copy the trainer source.
COPY . .

# Make the virtual env available system-wide in the container.
ENV PATH="${TRAINER_PACKAGE_DIR}/.venv/bin:$PATH"
