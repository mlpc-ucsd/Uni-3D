FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ARG DOCKER_USER=default_user USER_HOME=/workspace

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y sudo git ninja-build libopenblas-dev libopenexr-dev && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    adduser --disabled-password --gecos '' --home $USER_HOME $DOCKER_USER && adduser $DOCKER_USER sudo && \
    chown -R $DOCKER_USER:$DOCKER_USER $USER_HOME

USER $DOCKER_USER

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" FORCE_CUDA=1

ADD --chown=$DOCKER_USER:$DOCKER_USER requirements.txt $USER_HOME/

ADD --chown=$DOCKER_USER:$DOCKER_USER uni_3d/modeling/pixel_decoder/ops $USER_HOME/ops

RUN pip install --no-cache-dir -r $USER_HOME/requirements.txt && \
    pip install --no-cache-dir git+https://github.com/NVIDIA/MinkowskiEngine.git --global-option=--force_cuda && \
    pip install --no-cache-dir $USER_HOME/ops
