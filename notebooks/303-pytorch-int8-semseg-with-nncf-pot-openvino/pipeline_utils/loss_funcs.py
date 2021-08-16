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

import torch
import torch.nn.functional as F
import torchvision


def downsample_labels(labels, target_size=None, downsample_factor=None):
    H = labels.size()[1]
    W = labels.size()[2]
    if target_size is None and downsample_factor is None:
        raise ValueError("Either target_size or downsample_factor must be specified")
    if target_size is not None and downsample_factor is not None:
        raise ValueError("Only one of the target_size and downsample_factor must be specified")

    if downsample_factor is None:
        h = target_size[0]
        w = target_size[1]
    else:
        h = H // downsample_factor
        w = W // downsample_factor
    ih = torch.linspace(0, H - 1, h).long()
    iw = torch.linspace(0, W - 1, w).long()

    return labels[:, ih[:, None], iw]


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



def cross_entropy_aux(inputs: dict, target: torch.Tensor, weight: list):
    # This criterion is only suitable for `inputs` produced by the models
    # adopting the torchvision.models.segmentation._utils._SimpleSegmentationModel
    # contract - `inputs` shall be dicts of feature maps with keys corresponding
    # to the classifier type (either "out" for the main classifier or "aux" for
    # the auxiliary one)
    losses = {}
    for name, x in inputs.items():
        losses[name] = F.cross_entropy(x, target,
                                       ignore_index=255,
                                       weight=weight)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def cross_entropy(inputs, target: torch.Tensor, weight: list):
    # This criterion will support both the usual models producing
    # tensors as outputs and the torchvision models producing dict
    # of tensors, but without taking aux loss into account.
    input_tensor = None
    if isinstance(inputs, dict):
        input_tensor = inputs['out']
    else:
        input_tensor = inputs

    return F.cross_entropy(input_tensor, target,
                           ignore_index=255,
                           weight=weight)


def cross_entropy_icnet(inputs, target: torch.Tensor, weight: list):
    losses = {}

    if isinstance(inputs, dict):
        # Training - received a dict with auxiliary classifier outputs which
        # are downsampled relative to the ground-truth labels and input image
        target_ds4 = downsample_labels(target, downsample_factor=4)
        target_ds8 = downsample_labels(target, downsample_factor=8)
        target_ds16 = downsample_labels(target, downsample_factor=16)

        losses['ds4'] = F.cross_entropy(inputs['ds4'], target_ds4,
                                        ignore_index=255,
                                        weight=weight)
        losses['ds8'] = F.cross_entropy(inputs['ds8'], target_ds8,
                                        ignore_index=255,
                                        weight=weight)
        losses['ds16'] = F.cross_entropy(inputs['ds16'], target_ds16,
                                         ignore_index=255,
                                         weight=weight)

        return losses['ds4'] + 0.4 * losses['ds8'] + 0.4 * losses['ds16']

    # Testing - received classifier outputs with the same resolution as
    # the input image
    return F.cross_entropy(inputs, target, ignore_index=255, weight=weight)


def do_model_specific_postprocessing(model_name, labels, model_outputs):
    # pylint:disable=no-member
    metric_outputs = model_outputs
    if model_name == 'unet':
        # UNet predicts center image crops
        outputs_size_hw = (model_outputs.size()[2], model_outputs.size()[3])
        labels = center_crop(labels, outputs_size_hw).contiguous()
        metric_outputs = model_outputs
    elif model_name == 'icnet':
        if isinstance(model_outputs, dict):
            # Training - received a dict with auxiliary classifier outputs which
            # are downsampled relative to the ground-truth labels and input image
            # Will only calculate metric for the highest-res (1/4 size)
            # output, upscaled to 1x size
            metric_outputs = F.interpolate(model_outputs['ds4'], scale_factor=4)
    elif model_name in torchvision.models.segmentation.__dict__:
        # Torchvision segmentation models may output a dict of labels
        if isinstance(model_outputs, dict):
            metric_outputs = model_outputs['out']
    return labels, model_outputs, metric_outputs
