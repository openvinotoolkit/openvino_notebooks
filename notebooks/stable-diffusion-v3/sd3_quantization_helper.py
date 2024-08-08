import gc
import time
from typing import Any, Dict, List
import datasets
import matplotlib.pyplot as plt
import nncf
import numpy as np
import openvino as ov
import torch
from tqdm.notebook import tqdm
from transformers import set_seed
from sd3_helper import MODEL_DIR, TRANSFORMER_PATH, TEXT_ENCODER_PATH, TEXT_ENCODER_2_PATH, TEXT_ENCODER_3_PATH, VAE_DECODER_PATH, init_pipeline

TRANSFORMER_INT8_PATH = MODEL_DIR / "transformer_int8.xml"
TEXT_ENCODER_INT4_PATH = MODEL_DIR / "text_encoder_int4.xml"
TEXT_ENCODER_2_INT4_PATH = MODEL_DIR / "text_encoder_2_int4.xml"
VAE_DECODER_INT4_PATH = MODEL_DIR / "vae_decoder_int4.xml"
TEXT_ENCODER_3_INT4_PATH = MODEL_DIR / "text_encoder_3_int4.xml"


negative_prompts = [
    "blurry unreal occluded",
    "low contrast disfigured uncentered mangled",
    "amateur out of frame low quality nsfw",
    "ugly underexposed jpeg artifacts",
    "low saturation disturbing content",
    "overexposed severe distortion",
    "amateur NSFW",
    "ugly mutilated out of frame disfigured",
]

set_seed(42)

core = ov.Core()


def disable_progress_bar(pipeline, disable=True):
    if not hasattr(pipeline, "_progress_bar_config"):
        pipeline._progress_bar_config = {"disable": disable}
    else:
        pipeline._progress_bar_config["disable"] = disable


class CompiledModelDecorator(ov.CompiledModel):
    def __init__(self, compiled_model: ov.CompiledModel, data_cache: List[Any] = None, keep_prob: float = 0.5):
        super().__init__(compiled_model)
        self.data_cache = data_cache if data_cache is not None else []
        self.keep_prob = keep_prob

    def __call__(self, *args, **kwargs):
        if np.random.rand() <= self.keep_prob:
            self.data_cache.append(*args)
        return super().__call__(*args, **kwargs)


def collect_calibration_data(ov_pipe, calibration_dataset_size: int, num_inference_steps: int, guidance_scale) -> List[Dict]:
    original_model = ov_pipe.transformer
    calibration_data = []
    ov_pipe.transformer = CompiledModelDecorator(original_model, calibration_data, keep_prob=1)
    disable_progress_bar(ov_pipe)

    dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True, streaming=True)
    size = int(calibration_dataset_size // num_inference_steps)
    dataset = dataset.shuffle(seed=42).take(size)

    # Run inference for data collection
    pbar = tqdm(total=calibration_dataset_size)
    for batch in dataset:
        prompt = batch["caption"]
        negative_prompt = np.random.choice(negative_prompts)
        ov_pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=512, width=512)
        if len(calibration_data) >= calibration_dataset_size:
            pbar.update(calibration_dataset_size - pbar.n)
            break
        pbar.update(len(calibration_data) - pbar.n)

    disable_progress_bar(ov_pipe, disable=False)
    ov_pipe.transformer = original_model
    return calibration_data


def compress_model(model_path, save_path, group_size=128, ratio=0.8):
    if not save_path.exists():
        print(f"{model_path.stem} compression started")
        print(f"Compression parameters:\n\tmode = {nncf.CompressWeightsMode.INT4_SYM}\n\tratio = {ratio}\n\tgroup_size = {group_size}")
        model = core.read_model(model_path)
        compressed_model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_SYM, ratio=ratio, group_size=group_size)
        ov.save_model(compressed_model, save_path)
        print(f"{model_path.stem} compression finished")
    print(f"Compressed {model_path.stem} can be found in {save_path}")


def visualize_results(orig_img, optimized_img):
    """
    Helper function for results visualization

    Parameters:
       orig_img (Image.Image): generated image using FP16 models
       optimized_img (Image.Image): generated image using quantized models
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    orig_title = "FP16 pipeline"
    control_title = "INT8 pipeline"
    figsize = (20, 20)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex="all", sharey="all")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(optimized_img))
    list_axes[0].set_title(orig_title, fontsize=15)
    list_axes[1].set_title(control_title, fontsize=15)

    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()


def compare_models_size():
    fp16_model_paths = [TRANSFORMER_PATH, TEXT_ENCODER_PATH, TEXT_ENCODER_2_PATH, TEXT_ENCODER_3_PATH, VAE_DECODER_PATH]
    optimized_models = [TRANSFORMER_INT8_PATH, TEXT_ENCODER_INT4_PATH, TEXT_ENCODER_2_INT4_PATH, TEXT_ENCODER_3_INT4_PATH, VAE_DECODER_INT4_PATH]

    for fp16_path, optimized_path in zip(fp16_model_paths, optimized_models):
        if not fp16_path.exists():
            continue
        fp16_ir_model_size = fp16_path.with_suffix(".bin").stat().st_size
        optimized_model_size = optimized_path.with_suffix(".bin").stat().st_size
        print(f"{fp16_path.stem} compression rate: {fp16_ir_model_size / optimized_model_size:.3f}")


def compress_models(group_size=128, ratio=0.8):
    compress_model(TEXT_ENCODER_PATH, TEXT_ENCODER_INT4_PATH, group_size, ratio)
    compress_model(TEXT_ENCODER_2_PATH, TEXT_ENCODER_2_INT4_PATH, group_size, ratio)
    compress_model(VAE_DECODER_PATH, VAE_DECODER_INT4_PATH, group_size, ratio)
    if TEXT_ENCODER_3_PATH.exists():
        compress_model(TEXT_ENCODER_3_PATH, TEXT_ENCODER_3_INT4_PATH, group_size, ratio)


def calculate_inference_time(pipeline, validation_data, num_inference_steps, guidance_scale):
    inference_time = []
    pipeline.set_progress_bar_config(disable=True)
    for prompt in validation_data:
        start = time.perf_counter()
        _ = pipeline(
            prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            generator=torch.Generator().manual_seed(141),
        ).images[0]
        end = time.perf_counter()
        delta = end - start
        inference_time.append(delta)
    return np.median(inference_time)


def compare_perf(models_dict, opt_models_dict, device, use_flash_lora, validation_size=5, text_encoder_3_dim=4096):
    validation_dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", streaming=True, trust_remote_code=True)
    validation_dataset = validation_dataset.take(validation_size)
    validation_data = [batch["caption"] for batch in validation_dataset]

    print("Load FP16 pipeline")
    ov_pipe = init_pipeline(models_dict, device, use_flash_lora, text_encoder_3_dim)
    fp_latency = calculate_inference_time(ov_pipe, validation_data, 20 if not use_flash_lora else 4, 5 if not use_flash_lora else 0)
    del ov_pipe
    gc.collect()
    print("Load Optimized pipeline")
    optimized_pipe = init_pipeline(opt_models_dict, device, use_flash_lora, text_encoder_3_dim)
    opt_latency = calculate_inference_time(optimized_pipe, validation_data, 20 if not use_flash_lora else 4, 5 if not use_flash_lora else 0)
    print(f"Performance speed-up: {fp_latency / opt_latency:.3f}")
