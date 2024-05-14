#!bin/bash

apt install python3.8
apt install python3.8-distutils

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
rm get-pip.py

pip install --editable ./

pip install torchvision tensorboardX timm pytorch_lightning transformers
