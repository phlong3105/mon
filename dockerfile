# Docker commands
# sudo docker build -t aic23_track4 .
# sudo docker run -it aic23_track4 bash
# sudo docker run --name aic23_track4 --gpus all -it aic23_track4
# sudo docker exec -it aic23_track4 bash
# sudo docker commit 8081ea015084 phlong/aic23_track4
# sudo docker login
# sudo docker push phlong/aic23_track4
# sudo docker run --name aic23_track4 --gpus all -it phlong/aic23_track4

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 
# FROM nvcr.io/nvidia/pytorch:23.03-py3

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar
RUN apt install -y ffmpeg

# Create working directory
RUN mkdir -p /mon

# Install pip packages
RUN alias python=python3
RUN python3 -m pip install --upgrade pip wheel

# Install requirements
COPY . /mon

# conda create --name mon python=3.10
# conda install python=3.10
# conda config --add channels conda-forge
# conda install cudatoolkit=11.8 cudnn git git-lfs openblas
# pip install --root-user-action=ignore poetry setuptools pynvml markdown mkdocs mkdocs-material mkdocstrings sphinx sphinx-paramlinks albumentations ffmpeg-python opencv-python opencv-contrib-python requests matplotlib tensorboard einops flopco-pytorch lightning>=2.0.0 pytorch-lightning==1.9.2 piqa pyiqa thop torch>=2.0.0 torchaudio torchmetrics torchvision ray[tune] filterpy numpy scikit-image scikit-learn scipy click multipledispatch protobuf pyhumps PyYAML typing-extensions validators xmltodict rich tabulate tqdm
# poetry config virtualenvs.create false
# poetry install --with dev
# pip install --root-user-action=ignore -U openmim
# mim install mmcv-full==1.7.0
# conda clean --a --y

# RUN chmod +x /mon/install.sh
# RUN conda init bash
# RUN bash -l /mon/install.sh

# Make workspace
WORKDIR /mon

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype

# Entrypoint
# CMD ["conda activate mon"]
