from typing import Any, Dict, List
import datasets
import time
import torch

from collections import deque
from tqdm.notebook import tqdm
from transformers import set_seed
import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
from PIL import Image

from pixart_helper import MODEL_DIR, TEXT_ENCODER_PATH, TRANSFORMER_OV_PATH, VAE_DECODER_PATH

set_seed(42)
NUM_INFERENCE_STEPS = 4
INT8_TRANSFORMER_OV_PATH = MODEL_DIR / "transformer_ir_int8.xml"
INT4_TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder_int4.xml"
INT4_VAE_DECODER_PATH = MODEL_DIR / "vae_decoder_int4.xml"

NEGATIVE_PROMPTS = [
    "blurry unreal occluded",
    "low contrast disfigured uncentered mangled",
    "amateur out of frame low quality nsfw",
    "ugly underexposed jpeg artifacts",
    "low saturation disturbing content",
    "overexposed severe distortion",
    "amateur NSFW",
    "ugly mutilated out of frame disfigured",
]


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


def collect_calibration_data(pipe: "PixArtAlphaPipeline", subset_size: int) -> List[Dict]:
    calibration_data = []
    ov_transformer_model = pipe.transformer.transformer
    pipe.transformer.transformer = CompiledModelDecorator(ov_transformer_model, calibration_data, keep_prob=1.0)
    disable_progress_bar(pipe)

    size = int(np.ceil(subset_size / NUM_INFERENCE_STEPS))
    dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True, streaming=True)
    dataset = dataset.shuffle(seed=42).take(size)

    # Run inference for data collection
    pbar = tqdm(total=subset_size)
    for batch in dataset:
        caption = batch["caption"]
        if len(caption) > pipe.tokenizer.model_max_length:
            continue
        negative_prompt = np.random.choice(NEGATIVE_PROMPTS)
        pipe(
            prompt=caption,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(42),
            negative_prompt=negative_prompt,
            height=256,
            width=256,
        )
        if len(calibration_data) >= subset_size:
            pbar.update(subset_size - pbar.n)
            break
        pbar.update(len(calibration_data) - pbar.n)

    pipe.transformer.transformer = ov_transformer_model
    disable_progress_bar(pipe, disable=False)

    return calibration_data


def get_operation_const_op(operation, const_port_id: int):
    node = operation.input_value(const_port_id).get_node()
    queue = deque([node])
    constant_node = None
    allowed_propagation_types_list = ["Convert", "FakeQuantize", "Reshape"]

    while len(queue) != 0:
        curr_node = queue.popleft()
        if curr_node.get_type_name() == "Constant":
            constant_node = curr_node
            break
        if len(curr_node.inputs()) == 0:
            break
        if curr_node.get_type_name() in allowed_propagation_types_list:
            queue.append(curr_node.input_value(0).get_node())

    return constant_node


def is_embedding(node) -> bool:
    allowed_types_list = ["f16", "f32", "f64"]
    const_port_id = 0
    input_tensor = node.input_value(const_port_id)
    if input_tensor.get_element_type().get_type_name() in allowed_types_list:
        const_node = get_operation_const_op(node, const_port_id)
        if const_node is not None:
            return True

    return False


def get_quantization_ignored_scope(model):
    ops_with_weights = []
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            constant_node_0 = get_operation_const_op(op, const_port_id=0)
            constant_node_1 = get_operation_const_op(op, const_port_id=1)
            if constant_node_0 or constant_node_1:
                ops_with_weights.append(op.get_friendly_name())
        if op.get_type_name() == "Gather" and is_embedding(op):
            ops_with_weights.append(op.get_friendly_name())

    return ops_with_weights


def visualize_results(orig_img: Image, optimized_img: Image):
    """
    Helper function for results visualization

    Parameters:
       orig_img (Image.Image): generated image using FP16 models
       optimized_img (Image.Image): generated image using quantized models
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    orig_title = "FP16 pipeline"
    control_title = "Optimized pipeline"
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
    return fig


def compare_models_size():
    fp16_model_paths = [TRANSFORMER_OV_PATH, TEXT_ENCODER_PATH, VAE_DECODER_PATH]
    optimized_models = [INT8_TRANSFORMER_OV_PATH, INT4_TEXT_ENCODER_PATH, INT4_VAE_DECODER_PATH]

    for fp16_path, optimized_path in zip(fp16_model_paths, optimized_models):
        if not fp16_path.exists():
            continue
        fp16_ir_model_size = fp16_path.with_suffix(".bin").stat().st_size
        optimized_model_size = optimized_path.with_suffix(".bin").stat().st_size
        print(f"{fp16_path.stem} compression rate: {fp16_ir_model_size / optimized_model_size:.3f}")


def calculate_inference_time(pipeline, validation_data):
    inference_time = []
    pipeline.set_progress_bar_config(disable=True)

    for caption in validation_data:
        negative_prompt = np.random.choice(NEGATIVE_PROMPTS)
        start = time.perf_counter()
        pipeline(
            caption,
            negative_prompt=negative_prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(42),
        )
        end = time.perf_counter()
        delta = end - start
        inference_time.append(delta)

    pipeline.set_progress_bar_config(disable=False)
    return np.median(inference_time)


def compare_perf(ov_pipe, optimized_pipe, validation_size=3):
    validation_dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", streaming=True, trust_remote_code=True)
    validation_dataset = validation_dataset.take(validation_size)
    validation_data = [batch["caption"] for batch in validation_dataset]

    fp_latency = calculate_inference_time(ov_pipe, validation_data)
    print(f"FP16 pipeline: {fp_latency:.3f} seconds")
    opt_latency = calculate_inference_time(optimized_pipe, validation_data)
    print(f"Optimized pipeline: {opt_latency:.3f} seconds")
    print(f"Performance speed-up: {fp_latency / opt_latency:.3f}")
