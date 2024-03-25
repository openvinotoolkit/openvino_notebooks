import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms

from modules.midas.utils import normalize_unit_range
import modules.midas.normalization as normalization

class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        for item in sample.keys():
            interpolation_method = self.__image_interpolation_method
            sample[item] = cv2.resize(
                sample[item],
                (width, height),
                interpolation=interpolation_method,
            )

        if self.__resize_target:

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], 
                    (width, height), 
                    interpolation=cv2.INTER_NEAREST
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normalize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

class NormalizeIntermediate(object):
    """Normalize intermediate data by given mean and std.
    """

    def __init__(self, mean, std):

        self.__int_depth_mean = mean["int_depth"]
        self.__int_depth_std = std["int_depth"]

        self.__int_scales_mean = mean["int_scales"]
        self.__int_scales_std = std["int_scales"]

    def __call__(self, sample):

        if "int_depth" in sample and sample["int_depth"] is not None:
            sample["int_depth"] = (sample["int_depth"] - self.__int_depth_mean) / self.__int_depth_std

        if "int_scales" in sample and sample["int_scales"] is not None:
            sample["int_scales"] = (sample["int_scales"] - self.__int_scales_mean) / self.__int_scales_std

        return sample

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        for item in sample.keys():

            if sample[item] is None:
                pass
            elif item == "image":
                image = np.transpose(sample["image"], (2, 0, 1))
                sample["image"] = np.ascontiguousarray(image).astype(np.float32)
            else:
                array = sample[item].astype(np.float32)
                array = np.expand_dims(array, axis=0) # add channel dim
                sample[item] = np.ascontiguousarray(array)

        return sample


class Tensorize(object):
    """Convert sample to tensor.
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        for item in sample.keys():

            if sample[item] is None:
                pass
            else:
                # before tensorizing, verify that data is clean
                assert not np.any(np.isnan(sample[item])) 
                sample[item] = torch.Tensor(sample[item])

        return sample


def get_transforms(depth_predictor, sparsifier, nsamples):

    image_mean_dict = {
        "dpt_beit_large_512"    : [0.5, 0.5, 0.5],
        "dpt_swin2_large_384"   : [0.5, 0.5, 0.5],
        "dpt_large"             : [0.5, 0.5, 0.5], 
        "dpt_hybrid"            : [0.5, 0.5, 0.5],
        "dpt_swin2_tiny_256"    : [0.5, 0.5, 0.5],
        "dpt_levit_224"         : [0.5, 0.5, 0.5],
        "midas_small"           : [0.485, 0.456, 0.406],
    }

    image_std_dict = {
        "dpt_beit_large_512"    : [0.5, 0.5, 0.5],
        "dpt_swin2_large_384"   : [0.5, 0.5, 0.5],
        "dpt_large"             : [0.5, 0.5, 0.5], 
        "dpt_hybrid"            : [0.5, 0.5, 0.5],
        "dpt_swin2_tiny_256"    : [0.5, 0.5, 0.5],
        "dpt_levit_224"         : [0.5, 0.5, 0.5],
        "midas_small"           : [0.229, 0.224, 0.225],
    }

    resize_method_dict = {
        "dpt_beit_large_512"    : "minimal", 
        "dpt_swin2_large_384"   : "minimal",
        "dpt_large"             : "minimal", 
        "dpt_hybrid"            : "minimal", 
        "dpt_swin2_tiny_256"    : "minimal",
        "dpt_levit_224"         : "minimal",
        "midas_small"           : "upper_bound",
    }

    resize_dict = {
        "dpt_beit_large_512"    : 384,
        "dpt_swin2_large_384"   : 384,
        "dpt_large"             : 384,
        "dpt_hybrid"            : 384,
        "dpt_swin2_tiny_256"    : 256,
        "dpt_levit_224"         : 224,
        "midas_small"           : 384,
    }

    keep_aspect_ratio = True
    if "swin2" in depth_predictor or "levit" in depth_predictor:
        keep_aspect_ratio = False

    depth_model_transform_steps = [
        Resize(
            width=resize_dict[depth_predictor],
            height=resize_dict[depth_predictor],
            resize_target=False,
            keep_aspect_ratio=keep_aspect_ratio,
            ensure_multiple_of=32,
            resize_method=resize_method_dict[depth_predictor],
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(
            mean=image_mean_dict[depth_predictor], 
            std=image_std_dict[depth_predictor]
        ),
        PrepareForNet(),
        Tensorize(),     
    ]

    sml_model_transform_steps = [
        Resize(
            width=384,
            height=384,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_method_dict["midas_small"],
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeIntermediate(
            mean=normalization.VOID_INTERMEDIATE[depth_predictor][f"{sparsifier}_{nsamples}"]["mean"], 
            std=normalization.VOID_INTERMEDIATE[depth_predictor][f"{sparsifier}_{nsamples}"]["std"],
        ),
        PrepareForNet(),
        Tensorize(),
    ]

    return {
        "depth_model" : transforms.Compose(depth_model_transform_steps),
        "sml_model"   : transforms.Compose(sml_model_transform_steps),
    }
