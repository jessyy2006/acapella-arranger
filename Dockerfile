# Hugging Face Spaces Docker runtime for Aca-Adapt.
#
# HF Spaces Docker SDK requirements:
#   * Listen on port 7860 (HF sets ``PORT=7860`` and routes the public
#     URL to it — hard-coded below as a fallback for local `docker
#     run` in case the env var isn't set).
#   * Run as a non-root user with UID 1000 on writable /home/user
#     — HF enforces this for container sandboxing.
#   * Cold start should be reasonably quick; pull the heavy model
#     weights at runtime (already wired up in ``src/app/main.py``
#     via ``ACA_ADAPT_HF_REPO_ID``) rather than baking them into
#     the image, which would make the build slow and the image huge.

FROM python:3.10-slim

# ffmpeg is required by librosa / demucs / yt_dlp for any non-WAV
# input. Clean up apt cache in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root ``user`` with UID 1000, home
# directory at /home/user, working dir at /home/user/app.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache directory the app uses when pulling the checkpoint from
    # HF Hub. Making it explicit avoids the download landing in a
    # read-only location and failing silently.
    ACA_ADAPT_CACHE_DIR=/home/user/.cache/aca-adapt

WORKDIR /home/user/app

# Install Python deps first so a subsequent code-only change gets
# a warm Docker layer cache. Pinned torch CPU wheel — HF Spaces
# basic tier is CPU-only, and the default torch wheel pulls a
# ~900 MB CUDA variant that blows past the build timeout.
COPY --chown=user requirements.txt ./
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1 && \
    pip install --user -r requirements.txt

# Now copy the application source. Everything under the project
# root is fair game; .dockerignore trims anything we don't want
# shipped into the container.
COPY --chown=user . ./

EXPOSE 7860

# Streamlit wants ``--server.port`` and ``--server.address``
# explicitly to bind to the public interface HF Spaces expects.
# ``--server.headless true`` suppresses the "open browser" prompt;
# ``--server.enableCORS false`` is required for the HF Spaces
# iframe embed.
CMD ["sh", "-c", "streamlit run src/app/main.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --browser.gatherUsageStats false"]
