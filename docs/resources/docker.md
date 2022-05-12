<div align="center">
<p>
   <img height="200px" src="../../data/images/docker.png"></a>
</p>
<br>

Docker
=============================

<p align="center">
    <a href="https://github.com/phlong3105/one/blob/master/data/pdf/docker_cheatsheet.pdf">
        <img src="../../data/images/paper.png" height="32" alt="Cheatsheet"/>
    </a>
</p>
</div>

## Installation

<details open>
<summary><b style="font-size:18px">Docker</b></summary>

1. Uninstall (old) Docker engine. The contents of `/var/lib/docker/`, including images, containers, volumes, and networks, are preserved:
```shell
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get purge docker-ce docker-ce-cli containerd.io
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

2. Install using the repository:
```shell
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add stable repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install the latest version of Docker Engine and containerd
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

</details>

<details open>
<summary><b style="font-size:18px">NVIDIA Docker</b></summary>

<div align="center"><img width="480" src="../../data/images/nvidia_docker.png"></a></div>

```shell
# Setup the stable repository and the GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the nvidia-docker2 package (and dependencies) after updating the package listing:
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to complete the installation after setting the default runtime:
sudo systemctl restart docker

# At this point, a working setup can be tested by running a base CUDA container:
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

</details>
