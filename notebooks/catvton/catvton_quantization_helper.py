from typing import Any, List
from pathlib import Path
import pickle

from tqdm.notebook import tqdm
from transformers import set_seed
import numpy as np
import openvino as ov
from PIL import Image
import torch
import nncf

from ov_catvton_helper import (
    MODEL_DIR,
    VAE_ENCODER_PATH,
    VAE_DECODER_PATH,
    UNET_PATH,
    DENSEPOSE_PROCESSOR_PATH,
    SCHP_PROCESSOR_ATR,
    SCHP_PROCESSOR_LIP,
)

set_seed(42)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 2.5
GENERATOR = torch.Generator(device="cpu").manual_seed(42)

VAE_ENCODER_INT4_PATH = MODEL_DIR / "vae_encoder_int4.xml"
VAE_DECODER_INT4_PATH = MODEL_DIR / "vae_decoder_int4.xml"
UNET_INT8_PATH = MODEL_DIR / "unet_int8.xml"
DENSEPOSE_PROCESSOR_INT4_PATH = MODEL_DIR / "densepose_processor_int4.xml"
SCHP_PROCESSOR_ATR_INT4 = MODEL_DIR / "schp_processor_atr_int4.xml"
SCHP_PROCESSOR_LIP_INT4 = MODEL_DIR / "schp_processor_lip_int4.xml"


class CompiledModelDecorator(ov.CompiledModel):
    def __init__(
        self,
        compiled_model: ov.CompiledModel,
        data_cache: List[Any] = None,
        keep_prob: float = 1.0,
    ):
        super().__init__(compiled_model)
        self.data_cache = data_cache if data_cache is not None else []
        self.keep_prob = keep_prob

    def __call__(self, *args, **kwargs):
        if np.random.rand() <= self.keep_prob:
            self.data_cache.append(*args)
        return super().__call__(*args, **kwargs)


def collect_calibration_data(pipeline, automasker, mask_processor, dataset, subset_size):
    calibration_dataset_filepath = Path("calibration_data") / f"{subset_size}.pkl"
    calibration_dataset_filepath.parent.mkdir(exist_ok=True, parents=True)

    if not calibration_dataset_filepath.exists():
        original_unet = pipeline.unet.unet
        pipeline.unet.unet = CompiledModelDecorator(original_unet)

        calibration_dataset = []
        pbar = tqdm(total=subset_size, desc="Collecting calibration dataset")
        for data in dataset:
            person_image_path, cloth_image_path = data
            person_image = Image.open(person_image_path)
            cloth_image = Image.open(cloth_image_path)
            cloth_type = "upper" if "upper" in person_image_path.as_posix() else "overall"
            mask = automasker(person_image, cloth_type)["mask"]
            mask = mask_processor.blur(mask, blur_factor=9)

            pipeline(
                image=person_image,
                condition_image=cloth_image,
                mask=mask,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=GENERATOR,
            )
            collected_subset_size = len(pipeline.unet.unet.data_cache)
            pbar.update(NUM_INFERENCE_STEPS)
            if collected_subset_size >= subset_size:
                break

        calibration_dataset = pipeline.unet.unet.data_cache
        pipeline.unet.unet = original_unet

        with open(calibration_dataset_filepath, "wb") as f:
            pickle.dump(calibration_dataset, f)
    else:
        with open(calibration_dataset_filepath, "rb") as f:
            calibration_dataset = pickle.load(f)

    return calibration_dataset


def compress_model(core, model_path, save_path, group_size=128, ratio=0.8):
    if not save_path.exists():
        print(f"{model_path.stem} compression started")
        print(f"Compression parameters:\n\tmode = {nncf.CompressWeightsMode.INT4_SYM}\n\tratio = {ratio}\n\tgroup_size = {group_size}")
        model = core.read_model(model_path)
        compressed_model = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            ratio=ratio,
            group_size=group_size,
        )
        ov.save_model(compressed_model, save_path)
        print(f"{model_path.stem} compression finished")
    print(f"Compressed {model_path.stem} can be found in {save_path}")


def compress_models(core, group_size=128, ratio=0.8):
    compress_model(core, VAE_ENCODER_PATH, VAE_ENCODER_INT4_PATH, group_size, ratio)
    compress_model(core, VAE_DECODER_PATH, VAE_DECODER_INT4_PATH, group_size, ratio)
    compress_model(core, DENSEPOSE_PROCESSOR_PATH, DENSEPOSE_PROCESSOR_INT4_PATH, group_size, ratio)
    compress_model(core, SCHP_PROCESSOR_ATR, SCHP_PROCESSOR_ATR_INT4, group_size, ratio)
    compress_model(core, SCHP_PROCESSOR_LIP, SCHP_PROCESSOR_LIP_INT4, group_size, ratio)


def compare_models_size():
    fp16_model_paths = [
        VAE_ENCODER_PATH,
        VAE_DECODER_PATH,
        UNET_PATH,
        DENSEPOSE_PROCESSOR_PATH,
        SCHP_PROCESSOR_ATR,
        SCHP_PROCESSOR_LIP,
    ]
    optimized_models = [
        VAE_ENCODER_INT4_PATH,
        VAE_DECODER_INT4_PATH,
        UNET_INT8_PATH,
        DENSEPOSE_PROCESSOR_INT4_PATH,
        SCHP_PROCESSOR_ATR_INT4,
        SCHP_PROCESSOR_LIP_INT4,
    ]

    for fp16_path, optimized_path in zip(fp16_model_paths, optimized_models):
        if not fp16_path.exists():
            continue
        fp16_ir_model_size = fp16_path.with_suffix(".bin").stat().st_size
        optimized_model_size = optimized_path.with_suffix(".bin").stat().st_size
        print(f"{fp16_path.stem} compression rate: {fp16_ir_model_size / optimized_model_size:.3f}")
