ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME
ARG VARIANT
ARG DEPENDENCY_TAGS

FROM ghcr.io/${GIT_NAME}/ml-base:py${PYTHON_VERSION}-${VARIANT}

# Redefine ARGs to make them available after FROM
ARG PYTHON_VERSION
ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME
ARG VARIANT
ARG SSH_KEY_NAME
ARG DEPENDENCY_TAGS

WORKDIR /workspace/${PROJECT_NAME}

# Use system Python environment instead of virtual environment
ENV UV_SYSTEM_PYTHON=1
# Let uv automatically select PyTorch backend based on platform
ENV UV_TORCH_BACKEND=auto

# Setup Git configuration
RUN git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}"

# Copy only the scripts we need
COPY scripts/ray-init.sh /usr/local/bin/ray-init.sh
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/ray-init.sh /usr/local/bin/entrypoint.sh

# Copy only dependency files
COPY pyproject.toml uv.lock* ./

# Install everything to system Python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[${DEPENDENCY_TAGS}]" && \
    uv pip install --system ipdb

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["zsh"]
