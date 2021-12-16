FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ARG CODE_DIR="../../gfcn/"

# Changes thes sher to bash 
RUN chsh -s /bin/bash
ENV SHELL=/bin/bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install usefull tools
RUN apt-get -qy update && apt-get install -qy \
    man \
    vim \
    nano \
    htop \
    curl \
    wget \
    rsync \
    ca-certificates \
    git \
    zip \
    procps \
    ssh \
    gettext-base \
    transmission-cli \
    && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/*

# Installing the torch-geometric library
RUN pip install torch-geometric==1.3.1
RUN pip install torch-cluster==1.4.4  torch-scatter==1.3.1 torch-sparse==0.4.0 torch-spline-conv==1.1.0

# Installing the rest of the the stuff
RUN pip install keras==2.2.5
RUN pip install tensorflow==1.15.0
RUN pip install numpy==1.17.3
RUN pip install numpy-groupies==0.9.13
RUN pip install pillow==8.3.0
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install SimpleITK
RUN pip install nibabel
RUN pip install matplotlib
RUN pip install requests
RUN pip install torchfcn
RUN pip install numba
RUN pip install hausdorff

# Permision to run the OVHcloud user
COPY $CODE_DIR /workspace/gfcn
RUN chown -R 42420:42420 /workspace
WORKDIR /workspace/gfcn

ENTRYPOINT []
#CMD ["/workspace/gfcn/depth_training.sh 4ch 1"]
CMD ["python", "training.py", "-n", "GFCN", "-c", "BCElogistic"]
