# syntax=docker/dockerfile:1.7

# Two-stage build on Chainguard Python — distroless-style (Wolfi base, no
# shell, no package manager in the runtime image). Dependencies are
# installed with uv (Astral) in the builder; the runtime only needs the
# Python interpreter and the populated venv.
#
# Pin both stages to the same tag so the venv's Python ABI matches the
# runtime's Python ABI. `:latest` is fine for local builds; for
# production pin to a specific Python minor version
# (e.g. `python:3.13` / `python:3.13-dev`) or a digest.

ARG PY_IMAGE=cgr.dev/chainguard/python
ARG PY_TAG=latest
# Pinned to a specific patch — bump deliberately, not implicitly. For the
# tightest reproducibility, swap the tag for an image digest:
#   ghcr.io/astral-sh/uv@sha256:<digest>
ARG UV_VERSION=0.11.14

# ── uv image (aliased so the version ARG is resolved at FROM time) ─────────
# BuildKit expands ARGs in FROM but not reliably in `COPY --from=<image>`.
# Aliasing the uv image as its own stage sidesteps that quirk — the
# builder below copies from this stage by name.

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── builder ────────────────────────────────────────────────────────────────

FROM ${PY_IMAGE}:${PY_TAG}-dev AS builder

# Chainguard's -dev images run as nonroot by default. The builder needs
# to write to /opt/venv, so escalate here. The runtime stage stays
# nonroot — see USER nonroot below.
USER root

COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH

WORKDIR /src

# Layer the install so source edits don't bust the wheel cache. uv reads
# pyproject.toml directly; no requirements.txt needed.
COPY pyproject.toml ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python '.[postgres,serve]'

# Sanity check: surface import-time failures here, not at first request.
RUN python -c "import context_engine.server, context_engine.pg_store"

# ── runtime ────────────────────────────────────────────────────────────────

FROM ${PY_IMAGE}:${PY_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8080

COPY --from=builder /opt/venv /opt/venv

EXPOSE 8080

# Chainguard images default to a nonroot user; declaring it explicitly
# makes the policy obvious to anyone reading the file.
USER nonroot

# ENTRYPOINT is the Python interpreter, CMD picks the module. Lets a
# Fly scheduled Machine override CMD (e.g. `-m context_engine.brain_importer
# --source memories`) to run the importer instead of the server, sharing
# the same image and the CTX_PG_DSN secret.
#
# server.py reads HOST / PORT / LOG_LEVEL from env when invoked via the
# module entry point, so this CMD works on any platform that injects PORT
# (App Runner, Cloud Run, Heroku, Fargate-with-port-mapping).
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "context_engine.server"]

# ── follow-up notes ────────────────────────────────────────────────────────
#
# For byte-for-byte reproducible builds across CI runs, generate and
# commit a uv lockfile, then swap the install step for `uv sync`:
#
#   uv lock                          # produces uv.lock
#   git add uv.lock
#
# and replace the install RUN with:
#
#   COPY uv.lock ./
#   RUN --mount=type=cache,target=/root/.cache/uv \
#       uv sync --frozen --no-dev --extra postgres --extra serve \
#               --python /opt/venv/bin/python
#
# Held off here because the project doesn't yet have uv.lock checked in;
# adding it is a one-line follow-up.
