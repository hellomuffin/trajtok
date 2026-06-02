ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
# NOTE: Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y "ffmpeg>=6,<8" -c conda-forge && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.9.1
ARG TORCHAO_VERSION=0.15.0
ARG INSTALL_CHANNEL=whl
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchao==${TORCHAO_VERSION} torchvision torchaudio

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN TORCH_CUDA_ARCH_LIST="9.0 10.0" GROUPED_GEMM_CUTLASS="1" pip install --no-build-isolation --no-cache-dir "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# Install flash-attn 2
ARG FLASH_ATTN_VERSION=2.8.3
RUN FLASH_ATTN_CUDA_ARCHS="90;100" pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.4
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Install torchcodec.
ARG TORCH_CODEC_VERSION=0.9
RUN pip install --no-cache-dir torchcodec==${TORCH_CODEC_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY olmo/__init__.py olmo/__init__.py
COPY olmo/version.py olmo/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-molmo2 && \
    rm -rf *

# Install vllm.
ARG CUDA_VERSION_PATH
ARG VLLM_VERSION=0.15.1
RUN pip install vllm==${VLLM_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH}

# Install molmo-utils
RUN pip install --no-cache-dir "molmo-utils[torchcodec]"

# Install coco caption eval dependencies
RUN pip install --no-cache-dir pycocoevalcap
RUN /opt/conda/bin/conda install -y conda-forge::openjdk && /opt/conda/bin/conda clean -ya

# Install a few additional utilities via pip
RUN pip install --no-cache-dir \
    gpustat \
    jupyter \
    beaker-gantry

# Install torch-c-dlpack-ext
RUN pip install --no-cache-dir torch-c-dlpack-ext

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} AS release

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        rsync \
        vim \
        sudo \
        unzip \
        fish \
        parallel \
        zsh \
        htop \
        tmux \
        wget \
        emacs \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        apt-transport-https \
        gnupg \
        jq \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/* \
    # AWS CLI \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip \
    && rm -rf aws \
    # gsutil/gcloud \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && sudo apt-get update && sudo apt-get -y install google-cloud-cli \
    # GitHub CLI \
    && curl -sS https://webi.sh/gh | sh \
    # uv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Install DOCA OFED user-space drivers
# See https://docs.nvidia.com/doca/sdk/doca-host+installation+and+upgrade/index.html
# doca-ofed-userspace ver 2.10.0 depends on mft=4.31.0-149
ENV MFT_VER=4.31.0-149
RUN wget https://www.mellanox.com/downloads/MFT/mft-${MFT_VER}-x86_64-deb.tgz && \
    tar -xzf mft-${MFT_VER}-x86_64-deb.tgz && \
    mft-${MFT_VER}-x86_64-deb/install.sh --without-kernel && \
    rm mft-${MFT_VER}-x86_64-deb.tgz

ENV DOFED_VER=2.10.0
ENV OS_VER=ubuntu2404
RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOFED_VER}/host/doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    dpkg -i doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    apt-get update && apt-get -y install doca-ofed-userspace && \
    rm doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:/root/.cargo/bin:/opt/conda/bin:$PATH

# LABEL org.opencontainers.image.source https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core