#!/usr/bin/env bash
set -e
pip install -q -r requirements.txt
if [ ! -d third_party/sam2 ]; then
  git clone -q https://github.com/facebookresearch/sam2.git third_party/sam2
  git -C third_party/sam2 checkout -q 2b90b9f5ceec907a1c18123530e92e794ad901a4
fi
SAM2_BUILD_CUDA=0 pip install -q --no-deps -e third_party/sam2
mkdir -p weights
[ -f weights/sam2_hiera_base_plus.pt ] || wget -q -O weights/sam2_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
