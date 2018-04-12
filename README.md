# aitools
Jupyter Lab for local AI work with Pytorch, Tensorflow, OpenCV, OpenAI Gym (based on cudagl 9 image)

to build:
```
docker build --no-cache -t awokeknowing/aitools:18.4 -t awokeknowing/aitools:latest .
```

to get ready to run (may not be needed, so first try without):
```
xhost +si:localuser:root
```
to run (you need to have nvidia driver and nvidia-docker 2 aka runtime=nvidia installed):
```
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix awokeknowing/aitools
```


### setting up the host machine (ubuntu 16.04) will look something like:

```
sudo apt-get purge nvidia*
sudo apt install nvidia-390
sudo apt-get install cuda-drivers
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install     apt-transport-https     ca-certificates     curl     software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) \
     stable"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```
