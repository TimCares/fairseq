#!bin/bash

conda create -n fairseq python=3.8 ipython

conda activate fairseq

pip install --editable ./

pip install torchvision tensorboardX timm pytorch_lightning transformers