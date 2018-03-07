# aitools
pytorch with open ai gym, and keras/tensorflow and opencv (based on cudagl 9 image and floydhub/pytorch)

to build:
```
docker build --no-cache -t awokeknowing/aitools:2018-03-06 -t awokeknowing/aitools:latest .
```

to get ready to run (may not be needed, so first try without):
```
xhost +si:localuser:root
```
to run (you need to have nvidia driver and nvidia-docker 2 aka runtime=nvidia installed):
```
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix awokeknowing/aitools
```
