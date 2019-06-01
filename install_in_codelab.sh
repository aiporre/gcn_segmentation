#!/usr/bin/env bash
wget https://transfer.sh/3rm5B/data2.tar.xz -O data.tar.xz
tar xvf data.tar.xz
unzip data/M2NIST/combined.npy.zip -d data/M2NIST
unzip data/M2NIST/segmented.npy.zip -d data/M2NIST/
cd ../
git clone https://github.com/rusty1s/pytorch_geometric.git
cd ./pytorch_geometric
pip install --no-cache-dir torch-scatter
pip install --no-cache-dir torch-sparse
pip install --no-cache-dir torch-cluster
pip install --no-cache-dir torch-spline-conv
pip install torch-geometric
python3 example/gcn.py
cd ../gcn_air
pip install -r requirements.txt