import openvino as ov
import diffusers
import torch
import numpy as np

from collections import namedtuple
from typing import Tuple, Union
from pathlib import Path
from PIL import Image

import sys
sys.path.append("../utils")
from notebook_utils import download_file

class TextEncoder:
    """
    Text Encoder Adapter Class.

    This class is designed to seamlessly integrate the OpenVINO compiled model
    into the `stage_1.encode_prompt` routine.
    """

    def __init__(self, ir_path: Union[str, Path], dtype: torch.dtype, device: str = 'CPU') -> None:
        """
        Init the adapter with the IR model path.

        Parameters:
            ir_path (str, Path): text encoder IR model path
            dtype (torch.dtype): result dtype
            device (str): inference device
        Returns:
            None
        """
        self.ir_path = ir_path
        self.dtype = dtype
        self.encoder_openvino = ov.Core().compile_model(self.ir_path, device)

    def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None):
        """Adapt the network call."""
        result = self.encoder_openvino(input_ids)
        result_numpy = result[self.encoder_openvino.outputs[0]]
        return [torch.tensor(result_numpy, dtype=self.dtype)]


# The pipelines for Stages 1 and 2 expect the UNet models to return an object containing a sample attribute.
result_tuple = namedtuple('result', 'sample')


class UnetFirstStage:
    """
    IF Stage-1 Unet Adapter Class.

    This class is designed to seamlessly integrate the OpenVINO compiled model into
    the `stage_1` diffusion pipeline.
    """

    def __init__(self, unet_ir_path: Union[str, Path],
                 config: diffusers.configuration_utils.FrozenDict,
                 dtype: torch.dtype,
                 device: str = 'CPU'
                 ) -> None:
        """
        Init the adapter with the IR model path and model config.

        Parameters:
            unet_ir_path (str, Path): unet IR model path
            config (diffusers.configuration_utils.FrozenDict): original model config
            dtype (torch.dtype): result dtype
            device (str): inference device
        Returns:
            None
        """
        self.unet_openvino = ov.Core().compile_model(unet_ir_path, device)
        self.config = config
        self.dtype = dtype

    def __call__(self,
                 sample: torch.FloatTensor,
                 timestamp: int,
                 encoder_hidden_states: torch.Tensor,
                 class_labels: torch.Tensor = None,
                 cross_attention_kwargs: int = None,
                 return_dict: bool = False  # pipeline uses this argument when calling
                ) -> Tuple:
        """
        Adapt the network call.

        To learn more abould the model parameters please refer to
        its source code: https://github.com/huggingface/diffusers/blob/7200985eab7126801fffcf8251fd149c1cf1f291/src/diffusers/models/unet_2d_condition.py#L610
        """
        result = self.unet_openvino([sample, timestamp, encoder_hidden_states])
        result_numpy = result[self.unet_openvino.outputs[0]]
        return result_tuple(torch.tensor(result_numpy, dtype=self.dtype))


class UnetSecondStage:
    """
    IF Stage-2 Unet Adapter Class.

    This class is designed to seamlessly integrate the OpenVINO compiled model into
    the `stage_2` diffusion pipeline.
    """

    def __init__(self, unet_ir_path: Union[str, Path],
                 config: diffusers.configuration_utils.FrozenDict,
                 dtype: torch.dtype,
                 device: str = 'CPU'
                 ) -> None:
        """
        Init the adapter with the IR model path and model config.

        Parameters:
            unet_ir_path (str, Path): unet IR model path
            config (diffusers.configuration_utils.FrozenDict): original model config
            dtype (torch.dtype): result dtype
            device (str): inference device
        Returns:
            None
        """
        self.unet_openvino = ov.Core().compile_model(unet_ir_path, device)
        self.config = config
        self.dtype = dtype

    def __call__(self,
                 sample: torch.FloatTensor,
                 timestamp: int,
                 encoder_hidden_states: torch.Tensor,
                 class_labels: torch.Tensor = None,
                 cross_attention_kwargs: int = None,
                 return_dict: bool = False  # pipeline uses this argument when calling
                ) -> Tuple:
        """
        Adapt the network call.

        To learn more abould the model parameters please refer to
        its source code: https://github.com/huggingface/diffusers/blob/7200985eab7126801fffcf8251fd149c1cf1f291/src/diffusers/models/unet_2d_condition.py#L610
        """
        result = self.unet_openvino([sample, timestamp, encoder_hidden_states, class_labels])
        result_numpy = result[self.unet_openvino.outputs[0]]
        return result_tuple(torch.tensor(result_numpy, dtype=self.dtype))


def convert_result_to_image(result) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    result = 255 * result.squeeze(0).transpose(1, 2, 0)
    result[result < 0] = 0
    result[result > 255] = 255
    return Image.fromarray(result.astype(np.uint8), 'RGB')


def download_omz_model(model_name, models_dir):
    sr_model_xml_name = f'{model_name}.xml'
    sr_model_bin_name = f'{model_name}.bin'

    sr_model_xml_path = models_dir / sr_model_xml_name

    if not sr_model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        model_xml_url = base_url + sr_model_xml_name
        model_bin_url = base_url + sr_model_bin_name

        download_file(model_xml_url, sr_model_xml_name, models_dir)
        download_file(model_bin_url, sr_model_bin_name, models_dir)
    else:
        print(f'{model_name} already downloaded to {models_dir}')

