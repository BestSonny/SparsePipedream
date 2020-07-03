FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-extra freeglut3-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN rm -rf /opt/pytorch
RUN ldconfig
RUN git clone https://github.com/BestSonny/pytorch.git --recursive

# # Apply modifications and re-build PyTorch
RUN cd pytorch && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    python setup.py install

# # Reset default working directory
WORKDIR /workspace

ENV CUDA_HOME "/usr/local/cuda-10.2"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda-10.2/bin:${PATH}
ENV FORCE_CUDA=1

RUN pip install -r requirements.txt

RUN git clone https://github.com/BestSonny/MinkowskiEngineM.git && \
    cd MinkowskiEngineM && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    python setup.py install --force_cuda

WORKDIR /workspace
RUN git clone https://github.com/BestSonny/kaolin.git && \
    cd kaolin &&
    pip install -r requirements.txt
    python setup.py install

# # Reset default working directory
WORKDIR /workspace
