import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from typing import List
import logging as log
from pathlib import Path
from transformers.generation import GenerationConfig, GenerationMixin
import numpy as np

import openvino as ov
from openvino import opset13
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import PIL
from PIL import Image

from typing import Optional, Tuple, List, Union

from openvino import Core, Type
from openvino.passes import Manager, MatcherPass, WrapType, Matcher
from openvino import opset10 as ops
from openvino.preprocess import PrePostProcessor
import nncf

import time
import warnings
from transformers.utils.chat_template_utils import render_jinja_template
from image_processing_paddleocr_vl import PaddleOCRVLImageProcessor


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            root_output = matcher.get_match_value()
            root_name = root.get_friendly_name()
            if len(root.get_output_partial_shape(0)) == 3:
                parent = root.input_value(0).get_node()
                grand_parent = parent.input_value(0).get_node()
                grand_parent_output = parent.input(0).get_source_output()
                consumers = grand_parent_output.get_target_inputs()

                start = np.array([-1], dtype=np.int32)
                stop = np.array([-2], dtype=np.int32)
                step = np.array([-1], dtype=np.int32)
                axes = np.array([1], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")

                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)

                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)


class LlmStatefulModel:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        ov_model_path=None,
        device="CPU",
        fp16=False,
        int4_compress=False,
        int8_compress=False,
    ):
        self.name = "PaddleOCR_VL LLM Model"
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ov_model_path = ov_model_path
        self.fp16 = fp16
        self.int4_compress = int4_compress
        self.int8_compress = int8_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model.lm_head_module

    def get_input_names(self):
        inputs = ["attention_mask", "position_ids"]
        for idx in range(len(self.model.lm_head_module.decoder.layers)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        inputs.append("inputs_embeds")
        return inputs

    def get_output_names(self):
        outputs = ["logits"]
        for idx in range(len(self.model.lm_head_module.decoder.layers)):
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
        pass

    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f"tokenizer loading failed with {e}")

    def convert_sdpa_ov(self):
        llm_model = self.get_model()
        attention_mask = torch.ones(1, 213)

        llm_input = torch.rand((1, 213, 1024), dtype=torch.float32)
        positions_ids = torch.randint(
            low=0,
            high=33,
            size=(3, 1, attention_mask.shape[1]),
            dtype=torch.long,
        )
        pkv = llm_model(
            inputs_embeds=llm_input,
            attention_mask=attention_mask,
            position_ids=positions_ids,
            use_cache=True,
        )[1]

        llm_input = torch.rand((1, 1, 1024), dtype=torch.float32)
        attention_mask = torch.ones(1, 214)
        import numpy as np

        position_ids = torch.tensor([[[33]], [[33]], [[33]]], dtype=torch.long)  # shape: [3, 1, 1]

        llm_model.config.torchscript = True
        with torch.no_grad():
            ov_model = ov.convert_model(
                llm_model,
                example_input={
                    "inputs_embeds": llm_input,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": pkv,
                },
            )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model)
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(ov_model)

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_stateful.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_ASYM,
                "group_size": 64,
                "ratio": 0.9,
            }
            ov_compressed_model_int4 = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(
                ov_compressed_model_int4,
                Path(f"{self.ov_model_path}/llm_stateful_int4.xml"),
            )
            print(f"✅ INT4 compressed model saved to {self.ov_model_path}/llm_stateful_int4.xml")

        if self.int8_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT8_ASYM,
                # "group_size": 64,
                # "ratio": 1,
            }
            ov_compressed_model_int8 = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(
                ov_compressed_model_int8,
                Path(f"{self.ov_model_path}/llm_stateful_int8.xml"),
            )
            print(f"✅ INT8 compressed model saved to {self.ov_model_path}/llm_stateful_int8.xml")


class LlmEmbdModel:
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device="CPU",
        fp16=False,
    ):
        self.name = "PaddleOCR-VL Embd Model"
        self.model = model
        self.device = device
        self.ov_model_path = ov_model_path
        self.fp16 = fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.model.embed_tokens

    def get_input_names(self):
        inputs = ["input_ids"]
        return inputs

    def get_output_names(self):
        outputs = ["inputs_embeds"]
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
        pass

    def convert_sdpa_ov(self):
        embd_model = self.get_model()

        input_ids = torch.randint(0, 1000, (1, 213))

        ov_model = ov.convert_model(
            embd_model,
            example_input={
                "input": input_ids,
            },
        )

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_embd.xml"))


class VisionMlpModel:
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device="CPU",
        fp16=False,
    ):
        self.name = "Vision Mlp Model"
        self.model = model
        self.device = device
        self.ov_model_path = ov_model_path
        self.fp16 = fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.mlp_AR

    def get_input_names(self):
        return ["image_features", "image_grid_thw"]

    def get_output_names(self):
        outputs = ["vit_mlp"]
        return outputs

    def get_sample_input(self):
        pass

    def convert_sdpa_ov(self):
        encoder_model = self.get_model()
        # inputs_embeds = torch.rand(( 1, 800, 1152), dtype=torch.float32)
        # image_grid_thw = torch.tensor([[1, 20, 40]], dtype=torch.int32)
        inputs_embeds = torch.rand((1, 4988, 1152), dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 58, 86]], dtype=torch.int32)
        ov_model = ov.convert_model(
            encoder_model,
            example_input={
                "image_features": inputs_embeds,
                "image_grid_thw": image_grid_thw,
            },
        )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision_mlp.xml"))


import requests
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset
import tqdm


class VisionModel:
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device="CPU",
        fp16=False,
        int8_quant=False,
        tokenizer=None,
    ):
        self.name = "Vision Encoder Model"
        self.model = model
        self.device = device
        self.ov_model_path = ov_model_path
        self.fp16 = fp16
        self.inputs_dict = {}
        self.int8_quant = int8_quant
        self.tokenizer = tokenizer
        self.vision_pre_process = PaddleOCRVLPreprocessor(tokenizer=self.tokenizer)

    def get_model(self):
        return self.model.visual.vision_model

    def get_input_names(self):
        return ["pixel_values", "cu_seqlens", "image_grid_thw"]

    def get_output_names(self):
        outputs = ["vision_output"]
        return outputs

    def get_sample_input(self):
        pass

    def get_pil_from_path(self, image_path):
        """
        Loads an image from a local file path and converts it to a PIL Image object.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object in RGB format
        """
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((1200, 800), Image.Resampling.LANCZOS)
        return image

    def collate_fn(self, example, image_column="image_path"):
        """
        Preprocesses an example by loading and transforming image data from local file system.
        Loads the image specified by the path in the image_column.
        If there is any error during the loading process, returns None.
        Returns the preprocessed inputs with transformed image data.
        """
        assert len(example) == 1
        example = example[0]
        image_path = example[image_column]
        try:
            image = self.get_pil_from_path(image_path)
            h, w = image.size
            if h == 1 or w == 1:
                return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # Construct messages format expected by PaddleOCRVLPreprocessor.preprocess
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ""},
                ],
            }
        ]

        # Call preprocess with proper messages format
        inputs_dict = self.vision_pre_process.preprocess(messages=messages)

        # Return the result as a dictionary with images_info
        return {"images_info": inputs_dict["images_info"]}

    def prepare_calibration_data(self, dataloader, init_steps):
        """
        This function prepares calibration data from a dataloader for a specified number of initialization steps.
        It iterates over the dataloader, fetching batches and storing the relevant data.
        """
        data = []
        print(f"Fetching {init_steps} samples for the initialization...")
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch is not None:
                with torch.no_grad():
                    pixel_values = batch["images_info"]["pixel_values"]
                    image_grid_thw = batch["images_info"]["image_grid_thw"]

                    # Ensure pixel_values has batch dimension [1, N, 3, 14, 14]
                    original_shape = pixel_values.shape
                    if pixel_values.dim() == 4:  # Shape: (N, 3, 14, 14)
                        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension: [1, N, 3, 14, 14]
                    elif pixel_values.dim() == 5:
                        if pixel_values.shape[0] != 1:  # Shape: (B, N, 3, 14, 14) where B != 1
                            pixel_values = pixel_values[0:1]  # Take only first batch: [1, N, 3, 14, 14]
                    else:
                        raise ValueError(f"Unexpected pixel_values shape: {original_shape}, expected 4D or 5D tensor")

                    # Verify pixel_values has correct shape [1, N, 3, 14, 14]
                    if pixel_values.dim() != 5 or pixel_values.shape[0] != 1:
                        raise ValueError(f"pixel_values must have shape [1, N, 3, 14, 14], got {pixel_values.shape}")

                    # Ensure image_grid_thw has batch dimension [1, 3]
                    original_grid_shape = image_grid_thw.shape
                    if image_grid_thw.dim() == 1:  # Shape: (3,)
                        image_grid_thw = image_grid_thw.unsqueeze(0)  # Add batch dimension: [1, 3]
                    elif image_grid_thw.dim() == 2:
                        if image_grid_thw.shape[0] != 1:  # Shape: (B, 3) where B != 1
                            image_grid_thw = image_grid_thw[0:1]  # Take only first batch: [1, 3]
                    else:
                        raise ValueError(f"Unexpected image_grid_thw shape: {original_grid_shape}, expected 1D or 2D tensor")

                    # Verify image_grid_thw has correct shape [1, 3]
                    if image_grid_thw.dim() != 2 or image_grid_thw.shape[0] != 1 or image_grid_thw.shape[1] != 3:
                        raise ValueError(f"image_grid_thw must have shape [1, 3], got {image_grid_thw.shape}")

                    # Calculate actual sequence length from pixel_values
                    actual_seq_len = pixel_values.shape[1]  # Get N from [1, N, 3, 14, 14]
                    cu_seqlens = torch.tensor([0, actual_seq_len], dtype=torch.int32)

                    # Convert to numpy arrays for NNCF (it expects numpy arrays)
                    # Ensure shapes are correct: [1, N, 3, 14, 14] for pixel_values, [1, 3] for image_grid_thw
                    pixel_values_np = pixel_values.cpu().numpy()
                    image_grid_thw_np = image_grid_thw.cpu().numpy()
                    cu_seqlens_np = cu_seqlens.cpu().numpy()

                    # Final verification before adding to data
                    if pixel_values_np.shape[0] != 1:
                        raise ValueError(f"pixel_values numpy array must have batch size 1, got shape {pixel_values_np.shape}")

                    data.append(
                        {
                            "pixel_values": pixel_values_np,
                            "image_grid_thw": image_grid_thw_np,
                            "cu_seqlens": cu_seqlens_np,
                        }
                    )
        return data

    def prepare_dataset(self, opt_init_steps=3, max_train_samples=20):
        """
        Prepares a vision dataset for quantization using local images from test_images directory.
        Reads images from subdirectories (ocr, table, chart, formula) in test_images folder.
        """
        import os
        import random

        # Base directory for test images
        base_dir = Path(__file__).parent / "test_images"

        def _ensure_minimal_test_images(root_dir: Path) -> None:
            """
            Create a minimal set of test images under root_dir so that calibration can proceed.

            This is a best-effort fallback for environments where `test_images/` is missing.
            It creates 1 PNG per task type (ocr/table/chart/formula) with simple visual content.
            """
            try:
                from PIL import Image, ImageDraw  # pillow is already a dependency
            except Exception:
                # If Pillow is not available, keep the original behavior (raise later).
                return

            templates = {
                "ocr": "OCR: Hello 123",
                "table": "Table: A | B | C",
                "chart": "Chart: (mock)",
                "formula": "Formula: E = mc^2",
            }

            for subdir, text in templates.items():
                out_dir = root_dir / subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "sample.png"
                if out_path.exists():
                    continue

                # Create a simple white canvas with some text and shapes.
                im = Image.new("RGB", (1280, 720), color=(255, 255, 255))
                d = ImageDraw.Draw(im)
                d.rectangle([40, 40, 1240, 680], outline=(0, 0, 0), width=3)
                d.text((70, 80), text, fill=(0, 0, 0))
                if subdir == "table":
                    # Draw a minimal 3x3 grid to mimic a table.
                    x0, y0, x1, y1 = 120, 160, 880, 520
                    d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
                    for i in range(1, 3):
                        d.line(
                            [x0, y0 + (y1 - y0) * i / 3, x1, y0 + (y1 - y0) * i / 3],
                            fill=(0, 0, 0),
                            width=2,
                        )
                        d.line(
                            [x0 + (x1 - x0) * i / 3, y0, x0 + (x1 - x0) * i / 3, y1],
                            fill=(0, 0, 0),
                            width=2,
                        )
                elif subdir == "chart":
                    # Draw simple axes and a polyline.
                    d.line([120, 560, 900, 560], fill=(0, 0, 0), width=3)
                    d.line([120, 560, 120, 220], fill=(0, 0, 0), width=3)
                    d.line(
                        [140, 520, 280, 430, 420, 470, 560, 360, 700, 410, 860, 300],
                        fill=(0, 102, 204),
                        width=4,
                    )
                elif subdir == "formula":
                    # Add a second line to simulate a formula block.
                    d.text((70, 140), "∫_0^1 x^2 dx = 1/3", fill=(0, 0, 0))

                try:
                    im.save(out_path)
                except Exception:  # nosec B110 - non-critical image save, caller validates results
                    # Ignore image write errors; the caller will raise if no images exist.
                    pass

        # Collect all image paths from subdirectories
        image_paths = []
        for subdir in ["ocr", "table", "chart", "formula"]:
            subdir_path = base_dir / subdir
            if subdir_path.exists():
                for img_file in subdir_path.glob("*.png"):
                    image_paths.append(str(img_file))
                for img_file in subdir_path.glob("*.jpg"):
                    image_paths.append(str(img_file))
                for img_file in subdir_path.glob("*.jpeg"):
                    image_paths.append(str(img_file))

        if not image_paths:
            # Create minimal test images and retry collection once.
            _ensure_minimal_test_images(base_dir)

            for subdir in ["ocr", "table", "chart", "formula"]:
                subdir_path = base_dir / subdir
                if subdir_path.exists():
                    for img_file in subdir_path.glob("*.png"):
                        image_paths.append(str(img_file))
                    for img_file in subdir_path.glob("*.jpg"):
                        image_paths.append(str(img_file))
                    for img_file in subdir_path.glob("*.jpeg"):
                        image_paths.append(str(img_file))

        if not image_paths:
            raise ValueError(f"No images found in {base_dir} subdirectories")

        # Limit the number of samples
        if len(image_paths) > max_train_samples:
            image_paths = random.sample(image_paths, max_train_samples)  # nosec B311 - random subset for ML calibration, not security

        # Create a simple dataset with image paths
        class LocalImageDataset:
            def __init__(self, image_paths):
                self.image_paths = image_paths

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                return {"image_path": self.image_paths[idx]}

        dataset = LocalImageDataset(image_paths)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=self.collate_fn, batch_size=1, pin_memory=True)

        calibration_data = self.prepare_calibration_data(dataloader, opt_init_steps)
        return calibration_data

    def convert_sdpa_ov(self):
        vision_model = self.get_model()
        vision_model.eval()

        pixel_values = torch.rand((1, 4988, 3, 14, 14), dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 58, 86]], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 4988], dtype=torch.int32)

        with torch.no_grad():
            ov_model = ov.convert_model(
                vision_model,
                example_input={
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "cu_seqlens": cu_seqlens,
                },
            )

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision.xml"))

        if self.int8_quant:

            calibration_data = self.prepare_dataset()
            calibration_dataset = nncf.Dataset(calibration_data)
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6),
            )
            ov.save_model(quantized_model, Path(f"{self.ov_model_path}/vision_int8.xml"))


class PaddleOCR_VL_OV:
    def __init__(
        self,
        pretrained_model_path=None,
        model=None,
        tokenizer=None,
        ov_model_path="/tmp/paddleocr_vl_ov/",  # nosec B108 - overridable default for converted model cache
        device="CPU",
        llm_int4_compress=False,
        llm_int8_compress=False,
        vision_int8_quant=False,
    ):

        if model is None and pretrained_model_path:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = llm_int4_compress
        self.int8_compress = llm_int8_compress
        self.int8_quant = vision_int8_quant
        self.vision_model = VisionModel(
            model=self.model,
            ov_model_path=ov_model_path,
            device=device,
            int8_quant=self.int8_quant,
            tokenizer=self.tokenizer,
        )

        self.llm_embed_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = LlmStatefulModel(
            model=self.model,
            tokenizer=self.tokenizer,
            ov_model_path=ov_model_path,
            device=device,
            int4_compress=self.int4_compress,
            int8_compress=self.int8_compress,
        )

    def export_paddleocr_vl_to_ov(self):
        self.vision_model.convert_sdpa_ov()
        self.llm_embed_model.convert_sdpa_ov()
        self.llm_stateful_model.convert_sdpa_ov()
        print("✅ PaddleOCR-VL model has been successfully converted to OpenVINO format.")

    def close(self):
        """
        Release all resources held by this instance.
        After calling this method, the instance should not be used anymore.
        """
        import gc
        import torch

        # Release sub-model instances (they may hold references to the main model)
        if hasattr(self, "vision_model"):
            if hasattr(self.vision_model, "model"):
                del self.vision_model.model
            if hasattr(self.vision_model, "tokenizer"):
                del self.vision_model.tokenizer
            if hasattr(self.vision_model, "vision_pre_process"):
                del self.vision_model.vision_pre_process
            del self.vision_model

        if hasattr(self, "llm_embed_model"):
            if hasattr(self.llm_embed_model, "model"):
                del self.llm_embed_model.model
            del self.llm_embed_model

        if hasattr(self, "llm_stateful_model"):
            if hasattr(self.llm_stateful_model, "model"):
                del self.llm_stateful_model.model
            if hasattr(self.llm_stateful_model, "tokenizer"):
                del self.llm_stateful_model.tokenizer
            if hasattr(self.llm_stateful_model, "vision_pre_process"):
                del self.llm_stateful_model.vision_pre_process
            del self.llm_stateful_model

        # Release main model and tokenizer
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection to free memory immediately
        gc.collect()


class PaddleOCRVLPreprocessor:
    """
    Preprocessor class for PaddleOCR-VL model.
    Handles message preprocessing, image processing, and tokenization.
    """

    def __init__(self, tokenizer):
        """
        Initialize the preprocessor.

        Args:
            tokenizer: Tokenizer instance for text tokenization
        """
        self.tokenizer = tokenizer

    def preprocess(
        self,
        messages: List[dict],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        image_processor_config: Optional[dict] = None,
    ) -> dict:
        """
        Preprocess messages and images for the model.

        Args:
            messages: List of conversation messages. Each message should have the format:
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": PIL.Image},
                            {"type": "text", "text": "..."}
                        ]
                    }
                ]
            chat_template: Optional Jinja2 chat template string. If None, will use the default template.
            add_generation_prompt: Whether to add generation prompt to the template.
            image_processor_config: Optional dictionary with image processor configuration.
                Default values:
                {
                    "resample": 3,
                    "rescale_factor": 0.00392156862745098,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "min_pixels": 147384,
                    "max_pixels": 2822400,
                    "patch_size": 14,
                    "temporal_patch_size": 1,
                    "merge_size": 2
                }

        Returns:
            Dictionary containing:
                - "text_inputs": Tokenized text inputs from tokenizer
                - "images_info": Processed image information dictionary
        """
        # Use default chat template if not provided
        if chat_template is None:
            chat_template = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not eos_token is defined -%}\n    {%- set eos_token = "</s>" -%}\n{%- endif -%}\n{%- if not image_token is defined -%}\n    {%- set image_token = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " -}}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "image" -%}\n                {{ image_token }}\n            {%- endif -%}\n        {%- endfor -%}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] }}\n            {%- endif -%}\n        {%- endfor -%}\n        {{ "\\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " -}}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] }}\n            {%- endif -%}\n        {%- endfor -%}\n        {{ eos_token -}}\n    {%- elif message["role"] == "system" -%}\n        {%- for content in message["content"] -%}\n            {%- if content["type"] == "text" -%}\n                {{ content["text"] + "\\n" }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}\n'

        # Render Jinja template to get text with placeholders
        text, generation_indices = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

        # Default image processor configuration
        default_image_processor_config = {
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "min_pixels": 147384,
            "max_pixels": 2822400,
            "patch_size": 14,
            "temporal_patch_size": 1,
            "merge_size": 2,
        }

        # Merge user config with defaults
        if image_processor_config:
            default_image_processor_config.update(image_processor_config)

        # Create image processor
        image_processor = PaddleOCRVLImageProcessor(**default_image_processor_config)

        # Extract images from messages
        images = []
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content.get("type") == "image" and "image" in content:
                        images.append(content["image"])

        # Process images
        images_info = image_processor(images=images, return_tensors="pt")

        # Replace image placeholders in text
        if not isinstance(text, list):
            text = [text]

        index = 0
        for i in range(len(text)):
            while "<|IMAGE_PLACEHOLDER|>" in text[i]:
                text[i] = text[i].replace(
                    "<|IMAGE_PLACEHOLDER|>",
                    "<|placeholder|>" * (images_info["image_grid_thw"][index].prod() // 2 // 2),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|IMAGE_PLACEHOLDER|>")

        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt")

        return {
            "text_inputs": text_inputs,
            "images_info": images_info,
        }


class OVPaddleOCRVLForCausalLM(GenerationMixin):
    _is_stateful = True

    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device="CPU",
        llm_int4_compress=False,
        llm_int8_compress=False,
        vision_int8_quant=False,
        llm_int8_quant=False,
        llm_infer_list=[],
        vision_infer=[],
    ):

        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.llm_int4_compress = llm_int4_compress
        self.llm_int8_compress = llm_int8_compress
        self.vision_int8_quant = vision_int8_quant
        self.llm_int8_quant = llm_int8_quant

        ov_config = {
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "64",  # 32
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": "",
        }

        if llm_int4_compress:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful_int4.xml")
        elif llm_int8_compress:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful_int8.xml")
        else:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful.xml")
        if llm_int8_quant:
            self.llm_compiled_model = core.compile_model(self.llm_model, device, config=ov_config)
        else:
            self.llm_compiled_model = core.compile_model(self.llm_model, device)

        self.llm_request = self.llm_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_compiled_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_compiled_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, device)
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()

        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        # Initialize preprocessor
        self.preprocessor = PaddleOCRVLPreprocessor(tokenizer=self.tokenizer)

        self.vision_model_init()

        self.llm_infer_list = llm_infer_list
        self.vision_infer = vision_infer

        self.rope_deltas = None

    def vision_model_init(self):
        if self.vision_int8_quant:
            self.vision_encoder_model = Path(f"{self.ov_model_path}/vision_int8.xml")
        else:
            self.vision_encoder_model = Path(f"{self.ov_model_path}/vision.xml")
        # self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device, config = {'INFERENCE_PRECISION_HINT': 'f32'})
        self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device)

        self.vision_encoder_request = self.vision_encoder_compiled_model.create_infer_request()

        # self.vision_pre_process = Preprocess()
        # self.vision_middle_process = Postprocess()

    def vision_encoder_run(self, pixel_values=None, image_grid_thw=None, cu_seqlens=None):
        inputs_dict = {}
        inputs_dict["pixel_values"] = pixel_values
        inputs_dict["image_grid_thw"] = image_grid_thw
        inputs_dict["cu_seqlens"] = cu_seqlens
        self.vision_encoder_request.start_async(inputs_dict, share_inputs=True)
        self.vision_encoder_request.wait()
        return torch.from_numpy(self.vision_encoder_request.get_tensor("vision_output").data)

    def vision_model(self, pixel_values, image_grid_thw):
        encoder_start = time.perf_counter()

        if pixel_values is not None:
            pixel_values = pixel_values.unsqueeze(0)
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]

            pro = 0
            # breakpoint()
            for idx, thw in enumerate(image_grid_thw):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                cu_seqlens.append(cu_seqlens[-1] + numel)

            siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(pixel_values.device)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
            sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)
            image_grid_hws = torch.tensor(image_grid_hws, dtype=torch.int64)

            vit_embeds = self.vision_encoder_run(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                cu_seqlens=cu_seqlens,
            )
            encoder_end = time.perf_counter()
            encoder_time = (encoder_end - encoder_start) * 1000
            self.vision_infer.append(encoder_time)

            return vit_embeds

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs["input_ids"] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs_dict = {}
        if past_key_values is not None:
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict["inputs_embeds"] = inputs_embeds
        else:
            self.past_len = 0
            self.llm_request.reset_state()
            inputs_dict["inputs_embeds"] = inputs_embeds

        inputs_dict["attention_mask"] = attention_mask
        inputs_dict["position_ids"] = position_ids

        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in self.input_names:
            inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        start = time.perf_counter()
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        past_key_values = ((),)
        self.past_len += inputs_dict["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("logits").data),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_length = self.past_len
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - self.past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.past_len < input_ids.shape[1]:
                input_ids = input_ids[:, self.past_len :]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        else:
            self.llm_infer_list.clear()

        if past_key_values is not None:
            position_ids = kwargs.get("position_ids", None)
            batch_size, seq_length = input_ids.shape
            delta = (self.past_len + self.rope_deltas).to(input_ids.device) if self.past_len is not None else 0

            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if self.past_len is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    if torch.is_tensor(second_per_grid_t):
                        second_per_grid_t = second_per_grid_t.detach().item()
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def chat(
        self,
        messages=None,
        chat_template=None,
        generation_config=None,
        image_processor_config=None,
        verbose=False,
    ):
        # Handle default generation_config
        if generation_config is None:
            generation_config = {
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_new_tokens": 1024,
                "do_sample": False,
            }

        inputs_dict = self.preprocessor.preprocess(
            messages=messages,
            chat_template=chat_template,
            image_processor_config=image_processor_config,
        )
        input_ids = inputs_dict["text_inputs"]["input_ids"]
        attention_mask = inputs_dict["text_inputs"]["attention_mask"]
        pixel_values = inputs_dict["images_info"]["pixel_values"]
        image_grid_thw = inputs_dict["images_info"]["image_grid_thw"]

        inputs_embeds = self.llm_embd_run(input_ids)
        image_embeds = self.vision_model(pixel_values, image_grid_thw)

        image_token_id = self.config.image_token_id
        n_image_tokens = (input_ids == image_token_id).sum().item()
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        elif isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

        mask = input_ids == image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if attention_mask.dim() == 1:
            valid_seq_len = attention_mask.sum().item()
        else:
            valid_seq_len = attention_mask[0].sum().item()

        cache_position = torch.arange(0, valid_seq_len, device=self.device, dtype=torch.long)

        position_ids = None
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            None,
            attention_mask,
        )
        self.rope_deltas = rope_deltas

        generation_output = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **generation_config,
        )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        return response, None
