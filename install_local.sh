#!/usr/bin/env bash
wget https://transfer.sh/3rm5B/data2.tar.xz -O data.tar.xz
tar xvf data.tar.xz
unzip data/M2NIST/combined.npy.zip -d data/M2NIST
unzip data/M2NIST/segmented.npy.zip -d data/M2NIST/
export CUDA=cu101
echo '---------'
echo '---------'
echo ' INSTALLING Pytorch-Geo [==>...]'
echo '---------'
echo '---------'

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric
echo '---------'
echo '---------'
echo ' INSTALLING Pytorch-Geo [  DONE   ]'
echo '---------'
echo '---------'


pip install -r requirements.txt