import shutil
import time
import os
from glob import glob
from functools import partial
import numpy as np
import nncf
from nncf import compress_weights, Dataset
from nncf.parameters import CompressWeightsMode, SensitivityMetric
from transformers import AutoTokenizer
from transformers import AutoConfig
import openvino as ov
import json
from optimum.intel.openvino import OVModelForCausalLM
from datasets import load_dataset
from dataclasses import dataclass
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.exporters import TasksManager
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from transformers import AutoProcessor, AutoConfig, GenerationConfig


from data_preprocessing import prepare_dataset_llm
from ov_mllama_helper import OVMLlamaForConditionalGeneration

def compress(model: OVMLlamaForConditionalGeneration, algo = CompressWeightsMode.INT4_ASYM, ratio = 1.0,
                 sm = SensitivityMetric.MAX_ACTIVATION_VARIANCE,
                 awq=True, scale_estimation=True,
                 lora=False, gptq=False, group_size=64, all_layers=True):
    postfix = f"llm_{algo}_r{ratio}_gs{group_size}_{sm}"
    if awq:
        postfix += "_awq"
    if scale_estimation:
        postfix += "_scale"
    if gptq:
        postfix += "_gptq"
    if lora:
        postfix += "_lora"
    if all_layers:
        postfix += "_all_layers"

    postfix = postfix.replace('.', '')
    dst_name = postfix+".xml"

    if os.path.exists(dst_name):
        shutil.rmtree(dst_name)

    start = time.perf_counter()
    dataset = prepare_dataset_llm(model, 64)

    nncf_dataset = Dataset(dataset)

    model.model = compress_weights(model.model, mode=algo,
                                    group_size=group_size,
                                    ratio=ratio,
                                    dataset=nncf_dataset,
                                    sensitivity_metric=sm,
                                    awq=awq,
                                    scale_estimation=scale_estimation,
                                    gptq=gptq, all_layers=all_layers
                                    )

    end = time.perf_counter()


    print("Time: ", end - start)
    print(dst_name)
    ov.save_model(model.model, dst_name)
    #model.save_pretrained(dst_name)



model_id = "Llama-3.2-11B-Vision-Instruct/OV"
ov_model = OVMLlamaForConditionalGeneration(model_id)
processor = AutoProcessor.from_pretrained(model_id)

compress(ov_model)
