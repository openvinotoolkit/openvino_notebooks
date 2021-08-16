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

import os
from collections import OrderedDict
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToPILImage
import torchvision.transforms as T


def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def pil_loader(data_path, label_path):
    """Loads a sample and label image given their path as PIL images.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.

    Returns the image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label


def remap(image, old_values, new_values):
    assert isinstance(image, (Image.Image, np.ndarray)), "image must be of type PIL.Image or numpy.ndarray"
    assert isinstance(new_values, tuple), "new_values must be of type tuple"
    assert isinstance(old_values, tuple), "old_values must be of type tuple"
    assert len(new_values) == len(
        old_values), "new_values and old_values must have the same length"

    # If image is a PIL.Image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Replace old values by the new ones
    tmp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            # See Pylint issue #2721
            # pylint: disable=unsupported-assignment-operation
            tmp[image == old] = new

    return Image.fromarray(tmp)


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for _, label in tqdm(dataloader):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Ignore out-of-bounds target labels
        valid_indices = np.where((flat_label >= 0) & (flat_label < num_classes))
        flat_label = flat_label[valid_indices]

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for _, label in tqdm(dataloader):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Ignore out-of-bounds target labels
        valid_indices = np.where((flat_label >= 0) & (flat_label < num_classes))
        flat_label = flat_label[valid_indices]

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median

    freq = class_count / total
    freq = np.nan_to_num(freq)  # Guard against 0/0 divisions
    med = np.median(freq)
    class_weights = np.nan_to_num(med / freq)

    return class_weights


class LongTensorToRGBPIL:
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2)).fill_(0)

        for index, (_, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return ToPILImage()(color_tensor)



def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    import matplotlib.pyplot as plt
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def label_to_color(label, class_encoding):
    label_to_rgb = T.Compose([
        LongTensorToRGBPIL(class_encoding),
        T.ToTensor()
    ])
    return batch_transform(label.cpu(), label_to_rgb)


def color_to_label(color_labels: Image, class_encoding: OrderedDict):
    color_labels = np.array(color_labels.convert('RGB'))
    # pylint: disable=unsubscriptable-object
    labels = np.zeros((color_labels.shape[0], color_labels.shape[1]), dtype=np.int64)

    red = color_labels[..., 0]
    green = color_labels[..., 1]
    blue = color_labels[..., 2]

    for index, (_, color) in enumerate(class_encoding.items()):
        target_area = (red == color[0]) & (green == color[1]) & (blue == color[2])
        labels[target_area] = index

    labels = Image.fromarray(labels.astype(np.uint8), mode='L')
    return labels

def show_ground_truth_vs_prediction(images, gt_labels, color_predictions, class_encoding):
    """Displays three grids of images. The top grid displays ``images``
        the middle grid - ``gt_labels`` and the bottom grid - ``labels``

        Keyword arguments:
        - images (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)
        - gt_labels (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)
        - labels (``Tensor``): a 4D mini-batch tensor of shape
        (B, C, H, W)

        """

    import matplotlib.pyplot as plt
    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    color_predictions = torchvision.utils.make_grid(color_predictions).numpy()

    color_gt_labels = label_to_color(gt_labels, class_encoding)
    color_gt_labels = torchvision.utils.make_grid(color_gt_labels).numpy()

    plt.subplots(2, 3, figsize=(15, 7))
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(color_gt_labels, (1, 2, 0)))
    ax3.imshow(np.transpose(color_predictions, (1, 2, 0)))

    color_names = [k for k, _ in class_encoding.items()]
    colors = [(v[0] / 255.0, v[1] / 255.0, v[2] / 255.0, 1.0) for _, v in class_encoding.items()]
    ones = np.ones(len(colors))
    ax4.bar(range(0, len(ones)), ones, color=colors, tick_label=color_names)

    plt.show()


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


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets
