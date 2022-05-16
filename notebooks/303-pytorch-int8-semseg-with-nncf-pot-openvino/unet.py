"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# UNet implementation from:
# https://github.com/jvanvugt/pytorch-unet

from pkg_resources import parse_version

import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
            self,
            input_size_hw,
            in_channels=3,
            n_classes=2,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_channels (int): number of input channels
            input_size_hw: a tuple of (height, width) of the input images
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm prior to layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        if (input_size_hw[0] % 2**(depth - 1)) or (input_size_hw[1] % 2**(depth - 1)):
            raise ValueError("UNet may only operate on input resolutions aligned to 2**(depth - 1)")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        if torch._C._get_tracing_state() and parse_version(torch.__version__) >= parse_version("1.1.0"):
            # While exporting, add extra post-processing layers into the graph
            # so that the model outputs class probabilities instead of class scores
            softmaxed = F.softmax(x, dim=1)
            return softmaxed
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


def center_crop(layer, target_size):
    if layer.dim() == 4:
        # Cropping feature maps
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
            ]

    # If dimension is not 4, assume that we are cropping ground truth labels
    assert layer.dim() == 3
    _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[
        :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
        ]


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        self.padding = padding

    def forward(self, x, bridge):
        up = self.up(x)
        if self.padding:
            out = torch.cat([up, bridge], 1)
        else:
            crop1 = center_crop(bridge, up.shape[2:])
            out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


