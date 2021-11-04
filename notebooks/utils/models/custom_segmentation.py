# SegmentationModel implementation based on
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/models

import cv2
import numpy as np
from os import PathLike
from models import model
from notebook_utils import segmentation_map_to_overlay


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class SegmentationModel(model.Model):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        colormap: np.ndarray = None,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=False,
    ):
        """
        Segmentation Model for use with Async Pipeline

        :param model_path: path to IR model .xml file
        :param colormap: array of shape (num_classes, 3) where colormap[i] contains the RGB color
            values for class i. Optional for binary segmentation, required for multiclass
        :param resize_shape: if specified, reshape the model to this shape
        :param sigmoid: if True, apply sigmoid to model result
        :param argmax: if True, apply argmax to model result
        :param rgb: set to True if the model expects RGB images as input
        """
        super().__init__(ie, model_path)

        self.sigmoid = sigmoid
        self.argmax = argmax
        self.rgb = rgb

        self.net = ie.read_network(model_path, model_path.with_suffix(".bin"))
        self.output_layer = next(iter(self.net.outputs))
        self.input_layer = next(iter(self.net.input_info))
        if resize_shape is not None:
            self.net.reshape({self.input_layer: resize_shape})
        self.image_height, self.image_width = self.net.input_info[
            self.input_layer
        ].tensor_desc.dims[2:]

        if colormap is None and self.net.outputs[self.output_layer].shape[1] == 1:
            self.colormap = np.array([[0, 0, 0], [0, 0, 255]])
        else:
            self.colormap = colormap
        if self.colormap is None:
            raise ValueError("Please provide a colormap for multiclass segmentation")

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.
        """
        meta = {}
        image = inputs[self.input_layer]
        meta["frame"] = image
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height))
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)
        return {self.input_layer: input_image}, meta

    def postprocess(self, outputs, preprocess_meta):
        """
        Convert raw network results into an RGB segmentation map with overlay
        """
        alpha = 0.4

        if preprocess_meta["frame"].shape[-1] == 3:
            rgb_frame = preprocess_meta["frame"]
            if self.rgb:
                # reverse color channels to convert to BGR
                rgb_frame = rgb_frame[:, :, (2, 1, 0)]
        else:
            # Create RGB image by repeating channels in one-channel image
            rgb_frame = np.repeat(np.expand_dims(preprocess_meta["frame"], -1), 3, 2)
        res = outputs[self.output_layer].squeeze()

        result_mask_ir = sigmoid(res) if self.sigmoid else res

        if self.argmax:
            result_mask_ir = np.argmax(res, axis=0).astype(np.uint8)
        else:
            result_mask_ir = result_mask_ir.round().astype(np.uint8)
        overlay = segmentation_map_to_overlay(
            rgb_frame, result_mask_ir, alpha, colormap=self.colormap
        )

        return overlay
