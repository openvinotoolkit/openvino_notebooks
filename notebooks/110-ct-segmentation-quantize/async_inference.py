import cv2
import numpy as np
from collections import deque
from os import PathLike
import threading
from omz_python.models import model as omz_model
from pathlib import Path


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class CTAsyncPipeline:
    def __init__(self, ie, model, plugin_config, device="CPU", max_num_requests=0):

        # Enable model cachine for GPU devices
        if "GPU" in device and "GPU" in ie.available_devices:
            cache_path = Path("model/model_cache")
            cache_path.mkdir(exist_ok=True)
            ie.set_config({"CACHE_DIR": str(cache_path)}, device_name="GPU")

        self.model = model
        self.exec_net = ie.load_network(
            network=self.model.net,
            device_name=device,
            config=plugin_config,
            num_requests=max_num_requests,
        )
        self.empty_requests = deque(self.exec_net.requests)
        self.completed_request_results = {}
        self.callback_exceptions = {}
        self.event = threading.Event()

    def inference_completion_callback(self, status, callback_args):
        request, id, meta, preprocessing_meta = callback_args
        try:
            if status != 0:
                raise RuntimeError("Infer Request has returned status code {}".format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.completed_request_results[id] = (raw_outputs, meta, preprocessing_meta)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def submit_data(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        if len(self.empty_requests) == 0:
            self.event.clear()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        request.set_completion_callback(
            py_callback=self.inference_completion_callback,
            py_data=(request, id, meta, preprocessing_meta),
        )
        request.async_infer(inputs=inputs)

    def get_raw_result(self, id):
        if id in self.completed_request_results:
            return self.completed_request_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta = result
            return self.model.postprocess(raw_result, preprocess_meta, meta), meta
        return None

    def is_ready(self):
        return len(self.empty_requests) != 0

    def has_completed_request(self):
        return len(self.completed_request_results) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()


class SegModel(omz_model.Model):
    def __init__(self, ie, model_path: PathLike, colormap: np.ndarray = None, resize_shape=None):
        """
        Segmentation Model for use with Async Pipeline

        :param model_path: path to IR model .xml file
        :param colormap: array of shape (num_classes, 3) where colormap[i] contains the
                         RGB color values for class i
        :param resize_shape: if specified, reshape the model to this shape
        """
        super().__init__(ie, model_path)

        self.net = ie.read_network(model_path, model_path.with_suffix(".bin"))
        self.output_layer = next(iter(self.net.outputs))
        self.input_layer = next(iter(self.net.input_info))
        if resize_shape is not None:
            self.net.reshape({self.input_layer: resize_shape})
        self.image_height, self.image_width = self.net.input_info[
            self.input_layer
        ].tensor_desc.dims[2:]
        self.lut = None
        if colormap is not None:
            colormap = colormap.astype(np.uint8).reshape(20, 1, 3)
            self.lut = np.zeros((256, 1, 3), dtype=np.uint8)
            self.lut[:20, :, :] += colormap

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.
        """
        meta = {}
        image = inputs[self.input_layer]
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height))
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)
        return {self.input_layer: input_image}, meta

    def postprocess(self, outputs, preprocess_meta, meta):
        """
        Convert raw network results into an RGB segmentation map with overlay
        """
        alpha = 0.4
        res = outputs[self.output_layer][0, 0, ::]

        result_mask_ir = sigmoid(res).round().astype(np.uint8)
        result_image_ir = np.repeat(np.expand_dims(result_mask_ir, -1), 3, axis=2)
        result_image_ir[:, :, 2] *= 255
        meta.setdefault("frame", result_image_ir)
        rgb_frame = np.repeat(np.expand_dims(meta["frame"], -1), 3, 2)

        overlay = cv2.addWeighted(result_image_ir, alpha, rgb_frame, 1 - alpha, 0)
        return overlay