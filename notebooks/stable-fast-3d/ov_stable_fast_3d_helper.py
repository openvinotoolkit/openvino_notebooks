from collections import namedtuple
import math
from pathlib import Path
import types
from typing import List, Optional

import torch
from torch import nn

import openvino as ov


def convert(model: torch.nn.Module, xml_path: str, example_input, model_name: str):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        print(f"⌛ {model_name} conversion started")
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            converted_model = ov.convert_model(model, example_input=example_input)
        ov.save_model(converted_model, xml_path, compress_to_fp16=False)

        # cleanup memory
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
        print(f"✅ {model_name} model conversion finished")
    else:
        print(f"✅ Found converted {model_name} model")


def convert_image_tokenizer(image_tokenizer, output_dir):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        height = height // self.config.patch_size
        width = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        scale_factor = (
            (
                height / math.sqrt(num_positions),
                width / math.sqrt(num_positions),
            ),
        )
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                float(height / math.sqrt(num_positions)),
                float(width / math.sqrt(num_positions)),
            ),
            mode="bicubic",
            align_corners=False,
        )
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    image_tokenizer.model.embeddings.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, image_tokenizer.model.embeddings)

    example_input = {
        "images": torch.rand([1, 1, 3, 512, 512], dtype=torch.float32),
        "modulation_cond": torch.rand([1, 1, 768], dtype=torch.float32),
    }
    convert(image_tokenizer, output_dir, example_input, "Image tokenizer")


def convert_tokenizer(tokenizer, output_dir):
    convert(tokenizer, output_dir, torch.tensor(1), "Tokenizer")


def convert_backbone(backbone, output_dir):
    example_input = {
        "hidden_states": torch.rand([1, 1024, 27648], dtype=torch.float32),
        "encoder_hidden_states": torch.rand([1, 1297, 1024], dtype=torch.float32),
    }

    convert(backbone, output_dir, example_input, "Backbone")


def convert_post_processor(post_processor, output_dir):
    convert(post_processor, output_dir, torch.rand([1, 3, 1024, 32, 32], dtype=torch.float32), "Post processor")


def convert_camera_embedder(camera_embedder, output_dir):
    class CameraEmbedderWrapper(torch.nn.Module):
        def __init__(self, camera_embedder):
            super().__init__()
            self.camera_embedder = camera_embedder

        def forward(self, rgb_cond, mask_cond, c2w_cond, intrinsic_cond, intrinsic_normed_cond):
            kwargs = {
                "rgb_cond": rgb_cond,
                "mask_cond": mask_cond,
                "c2w_cond": c2w_cond,
                "intrinsic_cond": intrinsic_cond,
                "intrinsic_normed_cond": intrinsic_normed_cond,
            }
            embedding = self.camera_embedder(**kwargs)

            return embedding

    example_input = {
        "rgb_cond": torch.rand([1, 1, 512, 512, 3], dtype=torch.float32),
        "mask_cond": torch.rand([1, 1, 512, 512, 1], dtype=torch.float32),
        "c2w_cond": torch.rand([1, 1, 1, 4, 4], dtype=torch.float32),
        "intrinsic_cond": torch.rand([1, 1, 1, 3, 3], dtype=torch.float32),
        "intrinsic_normed_cond": torch.rand([1, 1, 1, 3, 3], dtype=torch.float32),
    }
    convert(CameraEmbedderWrapper(camera_embedder), output_dir, example_input, "Camera embedder")


def convert_image_estimator(image_estimator, output_dir):
    class ImageEstimatorWrapper(torch.nn.Module):
        def __init__(self, image_estimator):
            super().__init__()
            self.image_estimator = image_estimator

        def forward(self, cond_image):
            outputs = self.image_estimator(cond_image)
            filtered_ouptuts = {}
            for k, v in outputs.items():
                if k.startswith("decoder_"):
                    filtered_ouptuts[k] = v
            return filtered_ouptuts

    IMAGE_ESTIMATOR_OV_PATH = Path("models/image_estimator_ir.xml")
    example_input = {
        "cond_image": torch.rand([1, 1, 512, 512, 3], dtype=torch.float32),
    }
    convert(ImageEstimatorWrapper(image_estimator), output_dir, torch.rand([1, 1, 512, 512, 3], dtype=torch.float32), "Image estimator")


def convert_decoder(decoder, include_decoder_output_dir, exclude_decoder_output_dir):
    heads = [h for h in decoder.cfg.heads]
    include_cfg_decoder = [h for h in decoder.cfg.heads if h.name in ["vertex_offset", "density"]]
    exclude_cfg_decoder = [h for h in decoder.cfg.heads if h.name not in ["density", "vertex_offset"]]

    decoder.cfg.heads = include_cfg_decoder
    convert(decoder, include_decoder_output_dir, torch.rand([1, 535882, 120], dtype=torch.float32), "Decoder with include list")

    decoder.cfg.heads = exclude_cfg_decoder
    convert(decoder, exclude_decoder_output_dir, torch.rand([263302, 120], dtype=torch.float32), "Decoder with exclude list")
    decoder.cfg.heads = heads


class ImageTokenizerWrapper(torch.nn.Module):
    def __init__(self, image_tokenizer):
        super().__init__()
        self.image_tokenizer = image_tokenizer

    def forward(self, images, modulation_cond):
        inputs = {
            "images": images,
            "modulation_cond": modulation_cond,
        }
        outs = self.image_tokenizer(inputs)[0]

        return torch.from_numpy(outs)


class TokenizerWrapper(torch.nn.Module):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.detokenize = model.detokenize

    def forward(self, batch_size):
        outs = self.tokenizer(batch_size)[0]

        return torch.from_numpy(outs)


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
        }

        outs = self.backbone(inputs)[0]

        return torch.from_numpy(outs)


class PostProcessorWrapper(torch.nn.Module):
    def __init__(self, post_processor):
        super().__init__()
        self.post_processor = post_processor

    def forward(self, triplanes):
        outs = self.post_processor(triplanes)[0]

        return torch.from_numpy(outs)


class CameraEmbedderWrapper(torch.nn.Module):
    def __init__(self, camera_embedder):
        super().__init__()
        self.camera_embedder = camera_embedder

    def forward(self, **kwargs):
        outs = self.camera_embedder(kwargs)[0]

        return torch.from_numpy(outs)


class ImageEstimatorWrapper(torch.nn.Module):
    def __init__(self, image_estimator):
        super().__init__()
        self.image_estimator = image_estimator

    def forward(self, cond_image):
        outs = self.image_estimator(cond_image)

        results = {}
        for k, v in outs.to_dict().items():
            results[k.names.pop()] = torch.from_numpy(v)
        return results


class DecoderWrapper(torch.nn.Module):
    def __init__(self, include_decoder, exclude_decoder, heads):
        super().__init__()
        self.include_decoder = include_decoder
        self.exclude_decoder = exclude_decoder
        self.head_names = {head.name for head in heads}

    def forward(self, x, include: Optional[List] = None, exclude: Optional[List] = None):
        if include is not None:
            outs = self.include_decoder(x)
        else:
            outs = self.exclude_decoder(x)
        results = {}
        for k, v in outs.to_dict().items():
            for name in k.names:
                if name in self.head_names:
                    results[name] = torch.from_numpy(v)
        return results


def get_compiled_model(
    model,
    device,
    image_tokenizer_ov_path,
    tokenizer_ov_path,
    backbone_ov_path,
    post_processor_ov_path,
    camera_embedder_ov_path,
    image_estimator_ov_path,
    include_decoder_ov_path,
    exclude_decoder_ov_path,
):
    core = ov.Core()

    compiled_image_tokenizer = core.compile_model(image_tokenizer_ov_path, device.value)
    compiled_tokenizer = core.compile_model(tokenizer_ov_path, device.value)
    compiled_backbone = core.compile_model(backbone_ov_path, device.value)
    compiled_post_processor = core.compile_model(post_processor_ov_path, device.value)
    compiled_camera_embedder = core.compile_model(camera_embedder_ov_path, device.value)
    compiled_image_estimator = core.compile_model(image_estimator_ov_path, device.value)
    compiled_include_decoder = core.compile_model(include_decoder_ov_path, device.value)
    compiled_exclude_decoder = core.compile_model(exclude_decoder_ov_path, device.value)

    model.image_tokenizer = ImageTokenizerWrapper(compiled_image_tokenizer)
    model.tokenizer = TokenizerWrapper(compiled_tokenizer, model.tokenizer)
    model.backbone = BackboneWrapper(compiled_backbone)
    model.post_processor = PostProcessorWrapper(compiled_post_processor)
    model.camera_embedder = CameraEmbedderWrapper(compiled_camera_embedder)
    model.image_estimator = ImageEstimatorWrapper(compiled_image_estimator)
    model.decoder = DecoderWrapper(compiled_include_decoder, compiled_exclude_decoder, model.decoder.cfg.heads)

    return model
