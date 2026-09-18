
## docker build -t fedimoss/ft:v0.1.3 .

# Devel base: building csrc needs g++/nvcc, and runtime kernel JIT needs nvcc on PATH.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

# Default Python package index -> Tsinghua PyPI mirror (covers everything that is
# not pinned to an explicit index in pyproject: transformers, triton, flashinfer...).
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Tsinghua apt mirror. x86_64 sources live at archive/security.ubuntu.com;
# ports.ubuntu.com only exists on ARM64 images (which this engine does not build on).
RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g; s@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list.d/ubuntu.sources

# uv from the PyPI mirror instead of astral.sh/GitHub: the pip wheel ships the
# same prebuilt static binary. --break-system-packages is fine inside an image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages uv

COPY . /opt/FreeToken

# pyproject pins torch to download.pytorch.org/whl/cu130 (explicit index, the PyPI
# mirror cannot serve it) -- redirect it to SJTU's mirror, which syncs the same
# wheel tree. sglang-kernel stays on docs.sglang.io/whl/cu130: no domestic mirror.
RUN sed -i 's|https://download.pytorch.org/whl/cu130|https://mirror.sjtu.edu.cn/pytorch-wheels/cu130|' /opt/FreeToken/pyproject.toml

# One RUN: build isolation pulls a second copy of torch into the uv cache; clean it
# and drop the source tree in the same layer so neither survives in the image.
# --break-system-packages: Ubuntu 24.04 marks its system python externally managed
# (PEP 668); safe to bypass inside a single-purpose image.
RUN cd /opt/FreeToken \
    && uv pip install --system --break-system-packages ".[accel]" \
    && uv cache clean \
    && cd / \
    && rm -rf /opt/FreeToken

EXPOSE 1919
ENTRYPOINT ["ft"]
CMD ["--help"]
