# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import random
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
from kornia import enhance


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size, device):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3,
            3,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.blur_v = nn.Conv2d(
            3,
            3,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias), self.blur_h, self.blur_v
        ).to(device=device)

    def __call__(self, img):
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        return img


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        x_pil = transforms.ToPILImage()(x)
        x_sol = ImageOps.solarize(x_pil)
        return transforms.ToTensor()(x_sol).unsqueeze_(0)


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, image_size: int, device: str):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        self.base_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * image_size), device=device),
                transforms.RandomHorizontalFlip(),
                normalize,
            ]
        )
        self.base_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * image_size), device=device),
                enhance.solarize,
                transforms.RandomHorizontalFlip(),
                normalize,
            ]
        )

    def __call__(self, anchor, pos):
        anchor = self.base_transform1(anchor)
        pos = self.base_transform2(pos)
        return anchor, pos

    def anchor_transform(self, anchor):
        return self.base_transform1(anchor)

    def pos_transform(self, pos):
        return self.base_transform2(pos)
