"""
 Copyright (C) 2020 Intel Corporation

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

import copy
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Set, List, Optional, Callable

import cv2

from custom_segmentation import Model

sys.path.append("../utils")
from notebook_utils import show_array

def show_live_inference(
    ie, image_paths: List, model: Model, device: str, reader: Optional[Callable] = None
):
    """
    Do inference of images listed in `image_paths` on `model` on the given `device` and show
    the results in real time in a Jupyter Notebook

    :param image_paths: List of image filenames to load
    :param model: Model instance for inference
    :param device: Name of device to perform inference on. For example: "CPU"
    :param reader: Image reader. Should return a numpy array with image data.
                   If None, cv2.imread will be used, with the cv2.IMREAD_UNCHANGED flag
    """
    display_handle = None
    next_frame_id = 0
    next_frame_id_to_show = 0

    input_layer = model.net.input(0)

    # Create asynchronous pipeline and print time it takes to load the model
    load_start_time = time.perf_counter()
    pipeline = AsyncPipeline(
        ie=ie, model=model, plugin_config={}, device=device, max_num_requests=0
    )
    load_end_time = time.perf_counter()

    # Perform asynchronous inference
    start_time = time.perf_counter()

    while next_frame_id < len(image_paths) - 1:
        results = pipeline.get_result(next_frame_id_to_show)

        if results:
            # Show next result from async pipeline
            result, meta = results
            display_handle = show_array(result, display_handle)
            next_frame_id_to_show += 1
        if pipeline.is_ready():
            # Submit new image to async pipeline
            image_path = image_paths[next_frame_id]
            if reader is None:
                image = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_UNCHANGED)
            else:
                image = reader(str(image_path))
            pipeline.submit_data(
                inputs={input_layer: image}, id=next_frame_id, meta={"frame": image}
            )
            del image
            next_frame_id += 1
        else:
            # If the pipeline is not ready yet and there are no results: wait
            pipeline.await_any()

    pipeline.await_all()

    # Show all frames that are in the pipeline after all images have been submitted
    while pipeline.has_completed_request():
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            result, meta = results
            display_handle = show_array(result, display_handle)
            next_frame_id_to_show += 1

    end_time = time.perf_counter()
    duration = end_time - start_time
    fps = len(image_paths) / duration
    print(f"Loaded model to {device} in {load_end_time-load_start_time:.2f} seconds.")
    print(f"Total time for {next_frame_id} frames: {duration:.2f} seconds, fps:{fps:.2f}")

    del pipeline.exec_net
    del pipeline


def parse_devices(device_string):
    colon_position = device_string.find(':')
    if colon_position != -1:
        device_type = device_string[:colon_position]
        if device_type == 'HETERO' or device_type == 'MULTI':
            comma_separated_devices = device_string[colon_position + 1:]
            devices = comma_separated_devices.split(',')
            for device in devices:
                parenthesis_position = device.find(':')
                if parenthesis_position != -1:
                    device = device[:parenthesis_position]
            return devices
    return (device_string,)


def parse_value_per_device(devices: Set[str], values_string: str)-> Dict[str, int]:
    """Format: <device1>:<value1>,<device2>:<value2> or just <value>"""
    values_string_upper = values_string.upper()
    result = {}
    device_value_strings = values_string_upper.split(',')
    for device_value_string in device_value_strings:
        device_value_list = device_value_string.split(':')
        if len(device_value_list) == 2:
            if device_value_list[0] in devices:
                result[device_value_list[0]] = int(device_value_list[1])
        elif len(device_value_list) == 1 and device_value_list[0] != '':
            for device in devices:
                result[device] = int(device_value_list[0])
        elif device_value_list[0] != '':
            raise RuntimeError(f'Unknown string format: {values_string}')
    return result


def get_user_config(flags_d: str, flags_nstreams: str, flags_nthreads: int)-> Dict[str, str]:
    config = {}

    devices = set(parse_devices(flags_d))

    device_nstreams = parse_value_per_device(devices, flags_nstreams)
    for device in devices:
        if device == 'CPU':  # CPU supports a few special performance-oriented keys
            # limit threading for CPU portion of inference
            if flags_nthreads:
                config['CPU_THREADS_NUM'] = str(flags_nthreads)

            config['CPU_BIND_THREAD'] = 'NO'

            # for CPU execution, more throughput-oriented execution via streams
            config['CPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'CPU_THROUGHPUT_AUTO'
        elif device == 'GPU':
            config['GPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'GPU_THROUGHPUT_AUTO'
            if 'MULTI' in flags_d and 'CPU' in devices:
                # multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config['GPU_PLUGIN_THROTTLE'] = '1'
    return config


class AsyncPipeline:
    def __init__(self, ie, model, plugin_config, device='CPU', max_num_requests=0):
        cache_path = Path("model_cache")
        cache_path.mkdir(exist_ok=True)
        # Enable model caching for GPU devices
        if "GPU" in device and "GPU" in ie.available_devices:
            ie.set_property(device_name="GPU", properties={"CACHE_DIR": str(cache_path)})

        self.model = model
        self.logger = logging.getLogger()

        self.logger.info('Loading network to {} plugin...'.format(device))
        self.exec_net = ie.compile_model(self.model.net, device, plugin_config)
        if max_num_requests == 0:
            max_num_requests = self.exec_net.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS') + 1
        self.requests = [self.exec_net.create_infer_request() for _ in range(max_num_requests)]
        self.empty_requests = deque(self.requests)
        self.completed_request_results = {}
        self.callback_exceptions = []
        self.event = threading.Event()

    def inference_completion_callback(self, callback_args):
        try:
            request, id, meta, preprocessing_meta = callback_args
            raw_outputs = {out.any_name: copy.deepcopy(res.data) for out, res in zip(request.model_outputs, request.output_tensors)}
            self.completed_request_results[id] = (raw_outputs, meta, preprocessing_meta)
            self.empty_requests.append(request)
        except Exception as e:
            print(e)
            self.callback_exceptions.append(e)
        self.event.set()

    def submit_data(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        if len(self.empty_requests) == 0:
            self.event.clear()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        request.set_callback(self.inference_completion_callback, (request, id, meta, preprocessing_meta))
        request.start_async(inputs=inputs)
        request.wait()

    def get_raw_result(self, id):
        if id in self.completed_request_results:
            return self.completed_request_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta = result
            return self.model.postprocess(raw_result, preprocess_meta), meta
        return None

    def is_ready(self):
        return len(self.empty_requests) != 0

    def has_completed_request(self):
        return len(self.completed_request_results) != 0

    def await_all(self):
        for request in self.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()
