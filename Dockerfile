FROM nvidia/cudagl:9.0-devel-ubuntu16.04
LABEL maintainer "jamesdavidmorris@gmail.com"

# to build:
#   docker build --no-cache -t awokeknowing/aitools:2018-04-03 -t awokeknowing/aitools:latest .

# to set up machine for run: 
#   once:         
#     host machine needs ubuntu, nvidia drivers, docker, nvidia runtime for docker
#   each session: 
#     >cd /my/project/folder     (will be mounted into container at /root/work/)
#     >xhost +si:localuser:root  (gives permission to docker to foward GUI windows)

# to run (on local machine):
#   docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p8888:8888 -p6001:6001 -v$(pwd):/root/work awokeknowing/aitools

# at prompt:
# /root>python testgym.py  (should show space invaders, ctrl-c to exit)
# /root>lab                (should start jupyter lab. Click the link and should open in browser)

# in Juptyer Lab:
#   click "demo.ipynb"     (demo showing gym atari, pytorch, matplotlib, opencv, tensorboard)


####dependencies

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sudo \
        ca-certificates \
        curl \
        wget \
        bzr \
        git \
        mercurial \
        openssh-client \
        subversion \
        procps \
        autoconf \
        automake \
        bzip2 \
        file \
        g++ \
        gcc \
        imagemagick \
        libbz2-dev \
        libc6-dev \
        libcurl4-openssl-dev \
        libdb-dev \
        libevent-dev \
        libffi-dev \
        libgdbm-dev \
        libgeoip-dev \
        libglib2.0-dev \
        libjpeg-dev \
        libkrb5-dev \
        liblzma-dev \
        libmagickcore-dev \
        libmagickwand-dev \
        libncurses-dev \
        libpng-dev \
        libpq-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        libwebp-dev \
        libxml2-dev \
        libxslt-dev \
        libyaml-dev \
        make \
        patch \
        xz-utils \
        zlib1g-dev \
        # https://lists.debian.org/debian-devel-announce/2016/09/msg00000.html
        $( \
        # if we use just "apt-cache show" here, it returns zero because "Can't select versions from package 'libmysqlclient-dev' as it is purely virtual", hence the pipe to grep
            if apt-cache show 'default-libmysqlclient-dev' 2>/dev/null | grep -q '^Version:'; then \
            echo 'default-libmysqlclient-dev'; \
            else \
            echo 'libmysqlclient-dev'; \
            fi \
        ) \
    && rm -rf /var/lib/apt/lists/*

# This updates the global environment for the root user
RUN echo "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" >> /etc/environment
RUN echo "LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH" >> /etc/environment





################ python
# From https://github.com/docker-library/python/blob/master/3.6./Dockerfile

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tcl \
        tk \
    && rm -rf /var/lib/apt/lists/*

ENV GPG_KEY 0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D
ENV PYTHON_VERSION 3.6.2

RUN set -ex \
    && buildDeps=' \
        dpkg-dev \
        tcl-dev \
        tk-dev \
        wget \
        ca-certificates \
    ' \
    && apt-get update \
        && apt-get install -y $buildDeps --no-install-recommends \
        && rm -rf /var/lib/apt/lists/* \
    \
    && wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
    && wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
    && export GNUPGHOME="$(mktemp -d)" \
    && gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
    && gpg --batch --verify python.tar.xz.asc python.tar.xz \
    && rm -r "$GNUPGHOME" python.tar.xz.asc \
    && mkdir -p /usr/src/python \
    && tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
    && rm python.tar.xz \
    \
    && cd /usr/src/python \
    && gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
    && ./configure \
        --build="$gnuArch" \
        --enable-loadable-sqlite-extensions \
        --enable-shared \
        --with-system-expat \
        --with-system-ffi \
        --without-ensurepip \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && find /usr/local -depth \
        \( \
            \( -type d -a -name test -o -name tests \) \
            -o \
            \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
        \) -exec rm -rf '{}' + \
    && rm -rf /usr/src/python ~/.cache
# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
    && { [ -e easy_install ] || ln -s easy_install-* easy_install; } \
    && ln -s idle3 idle \
    && ln -s pydoc3 pydoc \
    && ln -s python3 python \
    && ln -s python3-config python-config


# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 9.0.3

RUN set -ex; \
    \
    wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py'; \
    \
    python get-pip.py \
        --disable-pip-version-check \
        --no-cache-dir \
        "pip==$PYTHON_PIP_VERSION" \
    ; \
    pip --version; \
    \
    find /usr/local -depth \
        \( \
            \( -type d -a \( -name test -o -name tests \) \) \
            -o \
            \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
        \) -exec rm -rf '{}' +; \
    rm -f get-pip.py

# install "virtualenv", since the vast majority of users of this image will want it
RUN pip install --no-cache-dir virtualenv



##################dependencies

# Add Bazel distribution URI as a package source
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list \
    && curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

# Install some dependencies
RUN apt-get update && apt-get install -y \
        ant \
        apt-utils \
        bazel \
        bc \
        build-essential \
        cmake \
        default-jdk \
        doxygen \
        gfortran \
        golang \
        iptables \
        libav-tools \
        libboost-all-dev \
        libeigen3-dev \
        libfreetype6-dev \
        libhdf5-dev \
        libjpeg-turbo8-dev \
        liblcms2-dev \
        libopenblas-dev \
        liblapack-dev \
        libpng12-dev \
        libprotobuf-dev \
        libsdl2-dev \
        libpython3-dev \
        libtiff-dev \
        libtiff5-dev \
        libvncserver-dev \
        libzmq3-dev \
        nano \
        net-tools \
        openmpi-bin \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        python3-opengl \
        python3-tk \
        python-software-properties \
        rsync \
        software-properties-common \
        swig \
        unzip \
        vim \
        webp \
        xorg-dev \
        xvfb \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/* \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
    && update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install Git LFS
RUN apt-get update \
    && add-apt-repository ppa:git-core/ppa \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs \
    && git lfs install \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/*











RUN apt-get clean && apt-get update && apt-get install -y \
        build-essential \
        cmake \
        gcc \
        apt-utils \
        pkg-config \
        make \
        nasm \
        wget \
        unzip \
        git \
        ca-certificates \
        curl \
        vim \
        nano \
        python3 \
        python3-pip \
        python3-dev \
        python3-numpy \
        gfortran \
        libatlas-base-dev \
        libatlas-dev \
        libatlas3-base \
        libhdf5-dev \
        libfreetype6-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libxml2-dev \
        libxslt-dev \
        libav-tools \
        libavcodec-dev \
        libavformat-dev \
        libxvidcore-dev \
        libx264-dev \
        x264 \
        libdc1394-22-dev \
        libswscale-dev \
        libv4l-dev \
        libsdl2-dev \
        swig \
        libboost-program-options-dev \
        libboost-all-dev \
        libboost-python-dev \
        zlib1g-dev \
        libjasper-dev \
        libtbb2 \
        libtbb-dev \
        libgl1-mesa-glx \
        qt5-default \
        libqt5opengl5-dev \
        xvfb \
        xorg-dev \
        x11-apps \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# add ffmpeg with cuda support

RUN git clone --depth 1 --branch n3.4.1 https://github.com/ffmpeg/ffmpeg ffmpeg && \
    cd ffmpeg && \
    ./configure --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp \
                --extra-cflags=-I/usr/local/cuda/include \
                --extra-ldflags=-L/usr/local/cuda/lib64 \
                --prefix=/usr/local/ffmpeg --enable-shared --disable-static \
                --disable-manpages --disable-doc --disable-podpages && \
                make -j"$(nproc)" install && \
                ldconfig

ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-npp-9-0 && \
    rm -rf /var/lib/apt/lists/*

#ENTRYPOINT ["ffmpeg"]

#WORKDIR /tmp
#CMD ["-y", "-hwaccel", "cuvid", "-c:v", "h264_cuvid", "-vsync", "0", "-i", \
#     "http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4", \
#     "-vf", "scale_npp=1280:720", "-vcodec", "h264_nvenc", "-t", "00:02:00", "output.mp4"]
#



RUN pip --no-cache-dir install \
        Cython \
        h5py \
        ipykernel \
        jupyter \
        jupyterlab \
        matplotlib \
        numpy \
        cupy \
        pandas \
        path.py \
        pyyaml \
        scipy \
        six \
        sklearn \
        sympy \
        Pillow \
        zmq \
        && \
    python -m ipykernel.kernelspec



# Create jupyter config
RUN mkdir /root/.jupyter && cd /root/.jupyter/ && \
/bin/echo -e "import os"                                                  > jupyter_notebook_config.py && \
/bin/echo -e "from IPython.lib import passwd"                            >> jupyter_notebook_config.py && \
/bin/echo -e ""                                                          >> jupyter_notebook_config.py && \
/bin/echo -e "c.NotebookApp.ip = '*'"                                    >> jupyter_notebook_config.py && \
/bin/echo -e "c.NotebookApp.port = int(os.getenv('PORT', 8888))"         >> jupyter_notebook_config.py && \
/bin/echo -e "c.NotebookApp.open_browser = False"                        >> jupyter_notebook_config.py && \
/bin/echo -e "c.MultiKernelManager.default_kernel_name = 'python3'"      >> jupyter_notebook_config.py && \
/bin/echo -e ""                                                          >> jupyter_notebook_config.py && \
/bin/echo -e "# sets a password if PASSWORD is set in the environment"   >> jupyter_notebook_config.py && \
/bin/echo -e "if 'PASSWORD' in os.environ:"                              >> jupyter_notebook_config.py && \
/bin/echo -e "  c.NotebookApp.password = passwd(os.environ['PASSWORD'])" >> jupyter_notebook_config.py && \
/bin/echo -e "  del os.environ['PASSWORD']"                              >> jupyter_notebook_config.py 

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
RUN printf '#!/bin/bash\njupyter notebook "$@" --allow-root' > /usr/bin/notebook && chmod +x /usr/bin/notebook
RUN printf '#!/bin/bash\njupyter lab      "$@" --allow-root' > /usr/bin/lab      && chmod +x /usr/bin/lab


# IPython
EXPOSE 8888


############# opencv


ARG OPENCV_VERSION=3.4.0

RUN apt-get update && apt-get install -y \
        python-opencv \
        libavcodec-dev \
        libavformat-dev \
        libav-tools \
        libavresample-dev \
        libdc1394-22-dev \
        libgdal-dev \
        libgphoto2-dev \
        libgtk2.0-dev \
        libjasper-dev \
        liblapacke-dev \
        libopencore-amrnb-dev \
        libopencore-amrwb-dev \
        libopencv-dev \
        libopenexr-dev \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libtheora-dev \
        libv4l-dev \
        libvorbis-dev \
        libvtk6-dev \
        libx264-dev \
        libxine2-dev \
        libxvidcore-dev \
        qt5-default \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN cd ~/ && \
    git clone https://github.com/Itseez/opencv.git --branch ${OPENCV_VERSION} --single-branch && \
    git clone https://github.com/Itseez/opencv_contrib.git --branch ${OPENCV_VERSION} --single-branch && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -DWITH_QT=ON \
        -DWITH_OPENGL=ON \
        -D WITH_CUDA=ON \
        -D CUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -DFORCE_VTK=ON \
        -DWITH_TBB=ON \
        -DWITH_GDAL=ON \
        -DWITH_XINE=ON \
        -DBUILD_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
 # Remove the opencv folders to reduce image size
    rm -rf ~/opencv*


# embed a file to test opencv
RUN cd /root/ && \
/bin/echo -e "import cv2 as cv"                                   > test-opencv.py && \
/bin/echo -e "import numpy as np"                                >> test-opencv.py && \
/bin/echo -e "for n in range(360):"                              >> test-opencv.py && \
/bin/echo -e " i = np.zeros((400,400,3), dtype=np.uint8)"        >> test-opencv.py && \
/bin/echo -e " for a in (90,0,45,-45):"                          >> test-opencv.py && \
/bin/echo -e "  cv.ellipse(i,(int(n**1.05),n),(99,25),a+n**2,"   >> test-opencv.py && \
/bin/echo -e "             0,360,(255,0,0),2,8)"                 >> test-opencv.py && \
/bin/echo -e " cv.circle(i,(int(n**1.05),n),n%50,(0,0,255),-1,8)">> test-opencv.py && \
/bin/echo -e " i = cv.blur(i,(20,20))"                           >> test-opencv.py && \
/bin/echo -e " cv.imshow(\"OpenCV\",i)"                          >> test-opencv.py && \
/bin/echo -e " cv.moveWindow(\"OpenCV\",0,200)"                  >> test-opencv.py && \
/bin/echo -e " cv.waitKey(25)"                                   >> test-opencv.py && \
/bin/echo -e "cv.destroyAllWindows()"                            >> test-opencv.py




#### open ai

# Add Tensorboard
RUN apt-get update && apt-get install -y supervisor \
  && apt-get clean \
  && apt-get autoremove \
  && rm -rf /var/cache/apt/archives/* \
  && rm -rf /var/lib/apt/lists/*
RUN printf '[program:tensorboard]\ncommand=tensorboard --port 6001 --logdir=/output' > /etc/supervisor/conf.d/tensorboard.conf
EXPOSE 6001

# graphviz for visualization
RUN apt-get update && apt-get install -y \
        graphviz \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*


RUN pip --no-cache-dir install \
        pydot \
        dlib \
        incremental \
        nltk \
        gym[atari,box2d,classic_control] \
        textacy \
        scikit-image \
        spacy \
        tqdm \
        wheel \
        kaggle-cli \
        annoy \
    && rm -rf /tmp/* /var/tmp/*


# Install OpenAI Universe
RUN git clone --branch v0.21.3 https://github.com/openai/universe.git \
    && cd universe \
    && pip install . \
    && cd .. \
    && rm -rf universe

# Install xgboost
RUN git clone --recursive https://github.com/dmlc/xgboost \
    && cd xgboost \
    && mkdir build \
    && cd build \
    && cmake .. -DUSE_CUDA=ON \
    && make -j$(nproc) \
    && cd .. \
    && cd python-package \
    && python setup.py install \
    && cd ../.. \
    && rm -rf xgboost





    ############## tensorflow and keras
RUN pip --no-cache-dir install tf-nightly-gpu

# Install Keras and tflearn
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION} \
        tflearn==0.3.2 \
    && rm -rf /pip_pkg \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache




###### pytorch 
RUN pip --no-cache-dir install torch tensorboardX torchvision


#### OpenAI gym
RUN apt-get update && apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig libgtk2.0-dev && git clone https://github.com/openai/gym.git && cd gym && pip install -e '.[classic_control,box2d,atari]' 

RUN printf "import gym\nenv = gym.make(\"SpaceInvaders-v0\")\nenv.reset()\nfor i in range(1500):\n  env.step(env.action_space.sample())\n  env.render()\nimport time\ntime.sleep(2)\nenv.close()" >> /root/test-gym.py

WORKDIR /root/

##### startup
RUN /bin/echo -e "#!/bin/bash\npython test-gym.py\necho \"\n\nWelcome to AI Lab by AwokeKnowing.\nType 'lab' to launch Jupyter Lab\"\n/bin/bash" > /usr/local/bin/onstartup.sh && chmod 777 /usr/local/bin/onstartup.sh

CMD /usr/local/bin/onstartup.sh
