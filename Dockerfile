
## docker build -t fedimoss/ft:v0.1.3 .

# Devel base: building csrc needs g++/nvcc, and runtime kernel JIT needs nvcc on PATH.
FROM nvidia/cuda:13.4.1-cudnn-devel-ubuntu24.04

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
#RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g; s@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list.d/ubuntu.sources

RUN sed -i \
    -e 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
    -e 's@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
    -e 's@http://ports.ubuntu.com/ubuntu-ports@https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/@g' \
    /etc/apt/sources.list.d/ubuntu.sources


# uv from the PyPI mirror instead of astral.sh/GitHub: the pip wheel ships the
# same prebuilt static binary. --break-system-packages is fine inside an image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages uv

COPY . /opt/FreeToken

# pyproject pins torch to download.pytorch.org/whl/cu130 (explicit index, the PyPI
# mirror cannot serve it) -- redirect it to SJTU's mirror, which syncs the same
# wheel tree. The sglang-cu130 index links its wheel to a github.com release asset
# (times out from CN); PyPI's sglang-kernel 0.4.5 is the same cu130 build, so drop
# the pin and let the Tsinghua PyPI mirror serve it instead.
RUN sed -i 's|https://download.pytorch.org/whl/cu130|https://mirror.sjtu.edu.cn/pytorch-wheels/cu130|' /opt/FreeToken/pyproject.toml \
    && sed -i '/sglang-kernel = { index = "sglang-cu130" }/d' /opt/FreeToken/pyproject.toml

# Keep the uv cache in a BuildKit cache mount: a timed-out build resumes without
# re-downloading torch, and the cache never lands in any image layer.
# --break-system-packages: Ubuntu 24.04 marks its system python externally managed
# (PEP 668); safe to bypass inside a single-purpose image.
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /opt/FreeToken \
    && uv pip install --system --break-system-packages ".[accel]" \
    ## https://github.com/NVIDIA/nccl/issues/2353   多卡卡死!!!
    && uv pip install --system --break-system-packages nvidia-nccl-cu13==2.31.2 --no-deps \
    && cd / \
    && rm -rf /opt/FreeToken

EXPOSE 1919
ENTRYPOINT ["ft"]
CMD ["--help"]
