FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y git ninja-build libopenblas-dev libopenexr-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" FORCE_CUDA=1

ADD requirements.txt /workspace/

ADD uni_3d/modeling/pixel_decoder/ops /workspace/ops

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/NVIDIA/MinkowskiEngine.git --global-option=--force_cuda

RUN cd ops && bash make.sh
