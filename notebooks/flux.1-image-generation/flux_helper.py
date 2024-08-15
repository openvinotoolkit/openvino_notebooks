import gc
import inspect
import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import types

import openvino as ov
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import numpy as np
import torch
from diffusers import FluxPipeline

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer


MODEL_DIR = Path("flux.1-shnell")

TRANSFORMER_PATH = Path("transformer/transformer.xml")
VAE_DECODER_PATH = Path("vae/vae_decoder.xml")
TEXT_ENCODER_PATH = Path("text_encoder/text_encoder.xml")
TEXT_ENCODER_2_PATH = Path("text_encoder_2/text_encoder_2.xml")


model_ids = ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]


def get_model_selector(default="black-forest-labs/FLUX.1-schnell"):
    import ipywidgets as widgets

    model_checkpoint = widgets.Dropdown(
        options=model_ids,
        default=default,
        description="Model:",
    )

    return model_checkpoint


def weight_compression_widget():
    import ipywidgets as widgets

    to_compress = widgets.Checkbox(
        value=True,
        description="Weight compression",
        disabled=False,
    )

    return to_compress


def get_pipeline_selection_option(opt_models_dict):
    import ipywidgets as widgets

    model_available = all([pth.exists() for pth in opt_models_dict.values()])
    use_quantized_models = widgets.Checkbox(
        value=model_available,
        description="Use compressed models",
        disabled=not model_available,
    )
    return use_quantized_models


def get_pipeline_components(model_dir, model_id="black-forest-labs/FLUX.1-schnell"):
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.tokenizer.save_pretrained(model_dir / "tokenizer")
    pipe.tokenizer_2.save_pretrained(model_dir / "tokenizer_2")
    pipe.scheduler.save_pretrained(model_dir / "scheduler")
    transformer, vae, text_encoder, text_encoder_2 = None, None, None, None
    if not (model_dir / TRANSFORMER_PATH).exists():
        transformer = pipe.transformer
        transformer.eval()
        transformer.save_config((model_dir / TRANSFORMER_PATH).parent)
    if not (model_dir / VAE_DECODER_PATH).exists():
        vae = pipe.vae
        vae.eval()
        vae.save_config((model_dir / VAE_DECODER_PATH).parent)
    if not (model_dir / TEXT_ENCODER_PATH).exists():
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        text_encoder.config.save_pretrained((model_dir / TEXT_ENCODER_PATH).parent)
    if not (model_dir / TEXT_ENCODER_2_PATH).exists():
        text_encoder_2 = pipe.text_encoder_2
        text_encoder_2.eval()
        text_encoder_2.config.save_pretrained((model_dir / TEXT_ENCODER_2_PATH).parent)
    return transformer, vae, text_encoder, text_encoder_2


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _prepare_latent_image_ids(batch_size, height, width, device=torch.device("cpu"), dtype=torch.float32):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels)

    return latent_image_ids.to(device=device, dtype=dtype)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


def _embednb_forward(self, ids: torch.Tensor) -> torch.Tensor:
    n_axes = ids.shape[-1]
    emb = torch.cat(
        [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
        dim=-3,
    )
    return emb.unsqueeze(1)


def convert_transformer(transformer, model_path):
    attention_dim = transformer.config.joint_attention_dim
    projection_dim = transformer.config.pooled_projection_dim
    transformer.forward = partial(transformer.forward, return_dict=False)
    transformer.pos_embed.forward = types.MethodType(_embednb_forward, transformer.pos_embed)
    __make_16bit_traceable(transformer)
    example_input = {
        "hidden_states": torch.zeros((1, 256, 64)),
        "timestep": torch.tensor([1], dtype=torch.float32),
        "encoder_hidden_states": torch.ones([1, 256, attention_dim]),
        "pooled_projections": torch.ones([1, projection_dim]),
        "txt_ids": torch.zeros([1, 256, 3]),
        "img_ids": _prepare_latent_image_ids(1, 32, 32),
    }
    if transformer.config.guidance_embeds:
        example_input["guidance"] = torch.tensor([5.0])

    with torch.no_grad():
        ov_model = ov.convert_model(
            transformer,
            example_input=example_input,
        )
    ov.save_model(ov_model, model_path)
    del ov_model
    cleanup_torchscript_cache()


def convert_t5_model(text_encoder, model_path):
    __make_16bit_traceable(text_encoder)
    with torch.no_grad():
        ov_model = ov.convert_model(text_encoder, example_input=torch.ones([1, 256], dtype=torch.long))
    ov.save_model(ov_model, model_path)
    del ov_model
    cleanup_torchscript_cache()


def convert_clip_model(text_encoder, text_encoder_path):
    text_encoder.forward = partial(text_encoder.forward, return_dict=False)
    __make_16bit_traceable(text_encoder)
    with torch.no_grad():
        ov_model = ov.convert_model(text_encoder, example_input=torch.ones([1, 77], dtype=torch.long))
    ov.save_model(ov_model, text_encoder_path)
    del ov_model
    cleanup_torchscript_cache()


def convert_vae_decoder(vae, model_path):
    __make_16bit_traceable(vae)
    with torch.no_grad():
        vae.forward = vae.decode
        ov_model = ov.convert_model(vae, example_input=torch.ones([1, vae.config.latent_channels, 64, 64]))
    ov.save_model(ov_model, model_path)
    del ov_model
    cleanup_torchscript_cache()


def convert_flux(model_id="black-forest-labs/FLUX.1-schnell"):
    model_dir = Path(model_id.split("/")[-1])
    conversion_statuses = [
        (model_dir / TRANSFORMER_PATH).exists(),
        (model_dir / VAE_DECODER_PATH).exists(),
        (model_dir / TEXT_ENCODER_PATH).exists(),
        (model_dir / TEXT_ENCODER_2_PATH).exists(),
    ]

    requires_conversion = not all(conversion_statuses)

    transformer, vae, text_encoder, text_encoder_2 = None, None, None, None

    if requires_conversion:
        transformer, vae, text_encoder, text_encoder_2 = get_pipeline_components(model_dir, model_id)
    else:
        print(f"✅ {model_id} model already converted and can be found in {model_dir}")
        return model_dir

    if not (model_dir / TRANSFORMER_PATH).exists():
        print("⌛ Transformer model conversion started")
        convert_transformer(transformer, model_dir / TRANSFORMER_PATH)
        del transformer
        gc.collect()
        print("✅ Transformer model conversion finished")

    else:
        print("✅ Found converted transformer model")

    if not (model_dir / TEXT_ENCODER_PATH).exists():
        print("⌛ Clip Text encoder conversion started")
        convert_clip_model(text_encoder, model_dir / TEXT_ENCODER_PATH)
        del text_encoder
        gc.collect()
        print("✅ Clip Text encoder conversion finished")
    else:
        print("✅ Found converted Clip Text encoder")

    if not (model_dir / TEXT_ENCODER_2_PATH).exists():
        print("⌛ T5 Text encoder conversion started")
        convert_t5_model(text_encoder_2, model_dir / TEXT_ENCODER_2_PATH)
        del text_encoder_2
        gc.collect()
        print("✅ T5 Text encoder conversion finished")
    else:
        print("✅ Found converted T5 Text encoder")

    if not (model_dir / VAE_DECODER_PATH).exists():
        print("⌛ VAE decoder conversion started")
        convert_vae_decoder(vae, model_dir / VAE_DECODER_PATH)
        del vae
        gc.collect()
        print("✅ VAE decoder onversion finished")
    else:
        print("✅ Found converted VAE decoder")

    print(f"✅ {model_id} successfully converted and can be found in {model_dir}")
    return model_dir


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OVFluxPipeline(DiffusionPipeline):
    def __init__(self, scheduler, transformer, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, transformer_config, vae_config):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_config = vae_config
        self.transformer_config = transformer_config
        self.vae_scale_factor = 2 ** (len(self.vae_config.get("block_out_channels", [0] * 16)) if hasattr(self, "vae") and self.vae is not None else 16)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        self.default_sample_size = 64

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = torch.from_numpy(self.text_encoder_2(text_input_ids)[0])

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = torch.from_numpy(self.text_encoder(text_input_ids)[1])

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width):
        return _prepare_latent_image_ids(batch_size, height, width)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width)
            return latents, latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer_config.get("in_channels", 64) // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer_config.get("guidance_embeds"):
                    guidance = torch.tensor([guidance_scale])
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None

                transformer_input = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "pooled_projections": pooled_prompt_embeds,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,
                    "img_ids": latent_image_ids,
                }
                if guidance is not None:
                    transformer_input["guidance"] = guidance

                noise_pred = torch.from_numpy(self.transformer(transformer_input)[0])

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents / self.vae_config.get("scaling_factor") + self.vae_config.get("shift_factor")
            image = self.vae(latents)[0]
            image = self.image_processor.postprocess(torch.from_numpy(image), output_type=output_type)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


def init_pipeline(model_dir, models_dict: Dict[str, Any], device: str):
    pipeline_args = {}

    print("Models compilation")
    core = ov.Core()
    for model_name, model_path in models_dict.items():
        pipeline_args[model_name] = core.compile_model(model_path, device)
        print(f"✅ {model_name} - Done!")

    transformer_path = models_dict["transformer"]
    transformer_config_path = transformer_path.parent / "config.json"
    with transformer_config_path.open("r") as f:
        transformer_config = json.load(f)
    vae_path = models_dict["vae"]
    vae_config_path = vae_path.parent / "config.json"
    with vae_config_path.open("r") as f:
        vae_config = json.load(f)

    pipeline_args["vae_config"] = vae_config
    pipeline_args["transformer_config"] = transformer_config

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler")

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(model_dir / "tokenizer_2")

    pipeline_args["scheduler"] = scheduler
    pipeline_args["tokenizer"] = tokenizer
    pipeline_args["tokenizer_2"] = tokenizer_2
    ov_pipe = OVFluxPipeline(**pipeline_args)
    return ov_pipe
