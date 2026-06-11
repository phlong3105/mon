# Using slim Debian base image - CUDA/cuDNN will come from PyTorch wheels
FROM debian:bookworm-slim

# Install uv python package manager and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy trainer files in the docker image
ENV TRAINER_PACKAGE_DIR=/opt/recipe
WORKDIR $TRAINER_PACKAGE_DIR

# Copy only dependency files first for caching
COPY pyproject.toml uv.lock .python-version README.md ./

# Install dependencies (cached if deps didn't change)
RUN uv sync --no-cache --locked

# Now copy the rest of the Trainer Package (source) files
COPY . .

# Adds the virtual environment in path to make it "system wide"
# Commands can be executed with out 'uv run': "uv run hafnia profile ls" -> "hafnia profile ls"
ENV PATH="${TRAINER_PACKAGE_DIR}/.venv/bin:$PATH"
