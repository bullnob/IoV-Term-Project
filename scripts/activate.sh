# Run with: source scripts/activate.sh
# Cloud snapshot activation; portable installations can use `uv run` instead.
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.toolchains/python
export UV_PROJECT_ENVIRONMENT=/workspace/.venvs/IoV-Term-Project
export UV_SYSTEM_CERTS=true
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
export PATH="/workspace/.toolchains/uv-0.13.6/bin:$PATH"
source /workspace/.venvs/IoV-Term-Project/bin/activate

