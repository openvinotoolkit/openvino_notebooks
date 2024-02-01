from collections import deque
import datasets
import pickle
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import StableDiffusionPipeline # DPMSolverMultistepScheduler
from transformers import set_seed

RANDOM_TEST_DATA = [
    "a black and brown dog standing outside a door.",
    "a person on a motorcycle makes a turn on the track.",
    "inflatable boats sit on the arizona river, and on the bank",
    "a white cat sitting under a white umbrella",
    "black bear standing in a field of grass under a tree.",
    "a train that is parked on tracks and has graffiti writing on it, with a mountain range in the background.",
    "a cake inside of a pan sitting in an oven.",
    "a table with paper plates and flowers in a home",
    "a photo of an astronaut riding a horse on mars",
    "Super cute fluffy cat warrior in armor, photorealistic, 4K, ultra detailed, vray rendering, unreal engine",
    "ultra close color photo portrait of rainbow owl with deer horns in the woods",
]


def dump_images(model_path, imgs_path):
    set_seed(1)
    core = ov.Core()
    model = core.read_model(model_path)

    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id = "/home/ltalamanova/cvs-117617/models/stable-diffusion-2-1"
    pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)
    # help(pipeline.unet)
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.model = model
    for i, prompt in enumerate(RANDOM_TEST_DATA):
        image = pipeline(prompt).images[0]
        image.save(f"{imgs_path}/{i}.png")


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

def collect_ops_with_weights(model):
    ops_with_weights = []
    for op in model.get_ops():
        # if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type(np.uint8):
        if op.get_type_name() == "MatMul":
            constant_node_0 = get_operation_const_op(op, const_port_id=0)
            constant_node_1 = get_operation_const_op(op, const_port_id=1)
            if constant_node_0 or constant_node_1:
                ops_with_weights.append(op.get_friendly_name())
        if op.get_type_name() == "Gather" and is_embedding(op):
            ops_with_weights.append(op.get_friendly_name())

    return ops_with_weights

class CompiledModelDecorator(ov.CompiledModel):
    def __init__(self, compiled_model, prob=0.5):
        super().__init__(compiled_model)
        self.data_cache = []
        self.prob = np.clip(prob, 0, 1)

    def __call__(self, *args, **kwargs):
        if np.random.rand() >= self.prob:
            self.data_cache.append(*args)
        return super().__call__(*args, **kwargs)


def collect_calibration_data(pipeline, subset_size=300):
    original_unet = pipeline.unet.model
    pipeline.unet.request = CompiledModelDecorator(ov.Core().compile_model(original_unet), prob=0.3)
    # pipeline.set_progress_bar_config(disable=True)

    dataset = datasets.load_dataset("conceptual_captions", split="train", streaming=True).shuffle(seed=42)

    for batch in dataset:
        prompt = batch["caption"]
        if len(prompt) > pipeline.tokenizer.model_max_length:
            continue
        _ = pipeline(prompt)
        collected_subset_size = len(pipeline.unet.request.data_cache)
        if collected_subset_size >= subset_size:
            break

    calibration_dataset = pipeline.unet.request.data_cache
    # pipeline.set_progress_bar_config(disable=False)
    pipeline.unet.model = original_unet
    pipeline.unet.request = None
    return calibration_dataset


def create_hybrid_model(model_id, model_path):
    pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)
    core = ov.Core()
    unet = core.read_model(model_path)
    subset_size = 300
    calibration_data = collect_calibration_data(pipeline, subset_size)
    print(f"Prepared {len(calibration_data)} calibration samples")
    quantization_dataset = nncf.Dataset(calibration_data)
    compressed_unet = nncf.compress_weights(unet)
    # compressed_unet = nncf.compress_weights(
    #     unet,
    #     dataset=quantization_dataset,
    #     mode=nncf.CompressWeightsMode.INT4_SYM,
    #     ratio=0.8,
    #     sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    # )
    unet_ignored_scope = collect_ops_with_weights(compressed_unet)
    print(f"Found {len(unet_ignored_scope)} nodes with compressed weights.")

    quantized_unet = nncf.quantize(
        model=compressed_unet,
        calibration_dataset=quantization_dataset,
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER,
        ignored_scope=nncf.IgnoredScope(names=unet_ignored_scope),
        advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=-1)
    )

    ov.save_model(quantized_unet, '/home/ltalamanova/cvs-117617/models/stable-diffusion-2-1/compressed/int8_asym_hybrid.xml')

# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "/home/ltalamanova/cvs-117617/models/stable-diffusion-2-1"
model_path = "/home/ltalamanova/cvs-117617/models/stable-diffusion-2-1/unet/openvino_model.xml"
create_hybrid_model(model_id, model_path)
