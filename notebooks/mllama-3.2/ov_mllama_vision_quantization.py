import logging
import nncf
import openvino as ov

from transformers import AutoProcessor
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from data_preprocessing import prepare_dataset_vision

processor = AutoProcessor.from_pretrained("Llama-3.2-11B-Vision-Instruct/OV/")
calibration_data = prepare_dataset_vision(processor, 100)
core = ov.Core()

nncf.set_log_level(logging.ERROR)
fp16_model_path = "Llama-3.2-11B-Vision-Instruct/OV/openvino_vision_encoder.xml"
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
