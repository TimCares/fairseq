# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging

import numpy as np
import os

import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL
import json

from fairseq.data import FairseqDataset
from .mae_image_dataset import caching_loader
from .imagenet_classes import IMAGENET2012_CLASSES

logger = logging.getLogger(__name__)


def build_transform(is_train, input_size, color_jitter, aa, reprob, remode, recount):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation="bicubic",
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class MaeFinetuningImageDataset(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        is_train: bool,
        input_size,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
        local_cache_path=None,
        shuffle=True,
    ):
        FairseqDataset.__init__(self)

        self.shuffle = shuffle

        transform = build_transform(
            is_train, input_size, color_jitter, aa, reprob, remode, recount
        )

        path = os.path.join(root, split)
        loader = caching_loader(local_cache_path, datasets.folder.default_loader)

        self.dataset = ImageNetDataset(data_path=path,
                                       split=split,
                                       loader=loader,
                                       transform=transform)

        logger.info(f"loaded {len(self.dataset)} examples")

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {"id": index, "image": img, "target": label}

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s["image"] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "image": collated_img,
            },
        }

        if "target" in samples[0]:
            res["net_input"]["target"] = torch.LongTensor([s["target"] for s in samples])

        return res

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]


class ImageNetDataset:
    def __init__(
            self,
            data_path:str,
            split,
            transform:torch.nn.Module,
            loader):
        self.data_path = data_path
        self.split = split
        self.path_to_data = os.path.join(self.data_path, 'imagenet')
        if not os.path.exists(self.path_to_data):
            raise FileNotFoundError(f"Directory {self.path_to_data} does not exists, "
                                    "please create it and add the correponding files from HuggingFace: "
                                    f"https://huggingface.co/datasets/imagenet-1k")
        
        self.path_to_split = os.path.join(self.path_to_data, self.split)
        os.makedirs(self.path_to_split, exist_ok=True)

        self.classes = {synset: i for i, synset in enumerate(IMAGENET2012_CLASSES.keys())}

        self.transform = transform
        self.loader = loader

    def load(self):
        items = []
        with open(os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl'), 'r', encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            self.log(f"Loaded {len(items)} {self.split} examples.")
        self.items = items

    def __len__(self):
        return len(self.items)

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    def __getitem__(self, index):
        item = self.items[index]
        image = self._get_image(image_path=item['image_path'])
        return image, item['target']
    
    def __len__(self):
        return len(self.items)
