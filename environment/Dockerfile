# hash:sha256:58621703e74c0dd8a2ec13a39ee72b05310441261d8af5de763cd02d884d88e5
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
    openpyxl==3.1.2 \
    pytorch-lightning==1.6.0 \
    pyyaml==6.0.1 \
    scikit-learn==1.3.0 \
    scipy==1.10.1 \
    seaborn==0.12.2 \
    torch==1.13.1 \
    wandb==0.15.11

COPY postInstall /
RUN /postInstall
