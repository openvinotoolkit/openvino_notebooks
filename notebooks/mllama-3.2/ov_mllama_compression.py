import shutil
import time
import os
from nncf import compress_weights, Dataset
from nncf.parameters import CompressWeightsMode, SensitivityMetric
import openvino as ov
from transformers import AutoProcessor
from pathlib import Path
import gc


from data_preprocessing import prepare_dataset_llm
from ov_mllama_helper import OVMLlamaForConditionalGeneration, LANGUAGE_MODEL

core = ov.Core()

def compress(model_dir: Path, processor:AutoProcessor, algo = CompressWeightsMode.INT4_ASYM, ratio = 1.0,
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
    nncf_dataset = None
    if awq or lora or gptq or scale_estimation:
        model = OVMLlamaForConditionalGeneration(model_dir, slice_lm_head=False)
        dataset = prepare_dataset_llm(model, processor, 64)
        nncf_dataset = Dataset(dataset)
        lm_model = model.model
        del model
        gc.collect()
    else:
        lm_model = core.read_model(model_dir / LANGUAGE_MODEL)

    lm_model = compress_weights(lm_model, mode=algo,
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
    saving_path = Path(model_dir) / dst_name
    ov.save_model(lm_model, saving_path)
    del lm_model
    gc.collect()
    return saving_path



model_id = "Llama-3.2-11B-Vision-Instruct/OV"
processor = AutoProcessor.from_pretrained(model_id)

compress(model_id, processor)
