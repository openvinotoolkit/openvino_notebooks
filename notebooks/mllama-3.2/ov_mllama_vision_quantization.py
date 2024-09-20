import torch
from datasets import load_dataset
from tqdm import tqdm

import logging
import nncf
import openvino as ov

import requests
from io import BytesIO
import numpy as np
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from data_preprocessing import prepare_dataset_vision

calibration_data = prepare_dataset_vision(100)
core = ov.Core()

nncf.set_log_level(logging.ERROR)
fp16_model_path = "Llama-3.2-11B-Vision-Instruct/OV/openvino_vision_encoder.xml"
fp16_model_path = "/home/aanuf/tmp/models/Meta-Llama-3.2-11B-Vision-Early/Llama-3.2-11B-Vision-Instruct/OV/openvino_vision_encoder.xml"
int8_model_path = fp16_model_path.replace('.xml', '_int8.xml')
ov_model = core.read_model(fp16_model_path)


calibration_dataset = nncf.Dataset(calibration_data)
quantized_model = nncf.quantize(
    model=ov_model,
    calibration_dataset=calibration_dataset,
    model_type=nncf.ModelType.TRANSFORMER,
    advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
)
ov.save_model(quantized_model, int8_model_path)
