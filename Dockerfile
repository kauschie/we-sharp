# Base image with CUDA and cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    openssh-client \
    p7zip-full \
    build-essential \
    tmux \
    rsync \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Add GitHub to known hosts (avoids SSH prompt)
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Install Miniconda
RUN curl -o /miniconda.sh -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set PATH and downgrade pip before creating env
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -y "pip<24.1" && conda clean -a -y

# Create Conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml --verbose && conda clean -a -y

# Activate environment
ENV CONDA_DEFAULT_ENV=audiolm_env
ENV PATH="/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH"

# Set working directory and copy files
WORKDIR /workspace
COPY . /workspace

# Install audiolm-pytorch manually without dependencies
RUN pip install audiolm-pytorch --no-deps

# Copy modified Python files into image
COPY lib_mods/*.py /tmp/lib_mods/

# Patch the modified files into the audiolm_pytorch site-packages
RUN cp /tmp/lib_mods/*.py /opt/conda/envs/audiolm_env/lib/python3.10/site-packages/audiolm_pytorch/.

# Entry point
CMD ["/bin/bash"]
