# Ubuntu 20.04 + Python 3.8 + CUDA 11.3.0 + cuDNN 8.2.0.41
FROM nvcr.io/nvidia/pytorch:21.04-py3

LABEL maintainer="Long Hoang Pham <longpham3105@gmail.com>"

# Adding folders
ADD data     /data/
ADD one      /one/
ADD projects /projects/

# Create `one` env
WORKDIR /one/install
RUN apt-get update -y
RUN apt-get install libgl1 -y
RUN chmod +x install_docker.sh
RUN conda init bash
RUN ./install_docker.sh

# Install `mish-cuda`
WORKDIR /one/install/mish-cuda
RUN conda init bash
RUN python setup.py build install

WORKDIR /one
RUN pip install --upgrade -e .

# Add projects
# WORKDIR /projects/aic
# RUN pip install --upgrade -e .

# WORKDIR /projects/chalearn
# RUN pip install --upgrade -e .

# WORKDIR /projects/ug2
# RUN pip install --upgrade -e .

# WORKDIR /projects/vipriors22
# RUN pip install --upgrade -e .

# NOTE: Back to root directory
WORKDIR /
ENTRYPOINT ["bin/bash"]
