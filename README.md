# About awokeknowing/aitools
To run you need to have nvidia driver and nvidia-docker 2 for runtime=nvidia to work.  For x-forwarding, eg to see matplatlib and gym atari games locallly, you may need to run `xhost +si:localuser:root` first):

`docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix awokeknowing/aitools`

### `awokeknowing/aitools` tagged `18.4` (latest) contain

- Based on new nVidia CUDA 9 with OpenGL image `nvidia/cudagl:9.0-devel-ubuntu16.04`
- python 3.6 (not anaconda)
- OpenCV 3.4 (compiled with cuda, blas)
- Tensorflow / Keras
- Pytorch (pip, not anaconda)
- OpenAI gym (with atari, classic control, and  box2d, not mujoco)
- Jupyter Lab (also Jupyter notebook)
- Various dependencies and extras related to the above (thanks floydhub)

Just type `lab` at the prompt when you run the image as mentioned above, with x-forwarding. Jupyter lab will start, and you can Ctrl+click the link to go to Jupyter Lab in your browser. I included a couple samples which if you run, will show OpenCV and OpenAI Gym running on your local display.

So as a final clarification, there is no 'desktop', nor browser in the image. It's just standard tools for working with AI on a cuda/ubuntu image. 

# building aitools
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
