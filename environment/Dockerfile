# hash:sha256:2626afff485bebab95075479a1e2ec0e8169a3c34e75cae104d8e709814776b3
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

COPY postInstall /
RUN /postInstall
