# hash:sha256:7782d3957ac5fb08f76405c9b8be3f5deac44fa653a3ed8a9894f160fbc553be
FROM registry.codeocean.allenneuraldynamics.org/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==3.5.3 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    h5py==3.9.0 \
    hydra-core==1.3.0 \
    matplotlib==3.7.2 \
    numpy==1.23.1 \
    pytorch-lightning==1.6.0 \
    pyyaml==6.0.1 \
    scikit-learn==1.3.0 \
    scipy==1.10.1 \
    seaborn==0.12.2 \
    torch==1.13.1 \
    wandb==0.15.11

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall
