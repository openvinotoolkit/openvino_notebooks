from pathlib import Path
import gc

import torch
import PIL.Image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, DiffusionPipeline, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from transformers import AutoTokenizer

import nncf
import openvino as ov

from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
import ftfy
import regex as re
import html


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


TEXT_ENCODER_PATH = "text_encoder.xml"
# IMAGE_ENCODER_PATH = "image_encoder.xml"
VAE_ENCODER_PATH = "vae_encoder.xml"
VAE_DECODER_PATH = "vae_decoder.xml"
TRANSFORMER_PATH = "transformer.xml"


def convert_pipeline(model_id, output_dir, compression_config=None):
    output_dir = Path(output_dir)

    required_paths = [TEXT_ENCODER_PATH, VAE_ENCODER_PATH, VAE_DECODER_PATH, TRANSFORMER_PATH]
    if all([(output_dir / model_path).exists() for model_path in required_paths]):
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")
        return

    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float32)

    transformer = pipe.transformer
    transformer.eval()
    vae = pipe.vae
    vae.eval()
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    # image_encoder = pipe.image_encoder
    # image_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    # image_processor = pipe.image_processor

    tokenizer.save_pretrained(output_dir / "tokenizer")
    scheduler.save_pretrained(output_dir / "scheduler")
    # image_processor.save_pretrained(output_dir / "image_processor")
    del pipe
    gc.collect()

    if not (output_dir / TEXT_ENCODER_PATH).exists():
        print("⌛ Convert Text Encoder model")
        __make_16bit_traceable(text_encoder)
        with torch.no_grad():
            text_encoder_inputs = {
                "input_ids": torch.ones((1, 512), dtype=torch.long),
                "attention_mask": torch.ones((1, 512), dtype=torch.long),
            }
            ov_model = ov.convert_model(text_encoder, example_input=text_encoder_inputs)
        if compression_config is not None:
            ov_model = nncf.compress_weights(ov_model, **compression_config)
        ov.save_model(ov_model, output_dir / TEXT_ENCODER_PATH)
        del ov_model
        cleanup_torchscript_cache()
        print(f"✅ Text Encoder successfully converted")
    del text_encoder
    gc.collect()

    if not (output_dir / VAE_ENCODER_PATH).exists():
        print("⌛ Convert VAE Encoder model")

        # Wrap encode to return tensor instead of AutoencoderKLOutput
        def vae_encode_wrapper(x):
            return vae.encode(x, return_dict=False)[0].mode()

        vae.forward = vae_encode_wrapper
        __make_16bit_traceable(vae)
        with torch.no_grad():
            ov_model = ov.convert_model(vae, example_input=torch.ones((1, 3, 1, 704, 544)))
        if compression_config is not None:
            ov_model = nncf.compress_weights(ov_model, **compression_config)
        ov.save_model(ov_model, output_dir / VAE_ENCODER_PATH)
        cleanup_torchscript_cache()
        print(f"✅ VAE Encoder successfully converted")

    if not (output_dir / VAE_DECODER_PATH).exists():
        print("⌛ Convert VAE Decoder model")
        vae.forward = vae.decode
        __make_16bit_traceable(vae)
        for up_block in vae.decoder.up_blocks:
            if up_block.upsampler is not None:
                up_block.upsampler.resample[0].mode = "nearest"
        with torch.no_grad():
            ov_model = ov.convert_model(vae, example_input=(torch.ones((1, 48, 6, 44, 34))))
        if compression_config is not None:
            ov_model = nncf.compress_weights(ov_model, **compression_config)
        ov.save_model(ov_model, output_dir / VAE_DECODER_PATH)
        cleanup_torchscript_cache()
        print(f"✅ VAE Decoder successfully converted")
    del vae
    gc.collect()
    print(f"✅ Model successfully converted and can be found in {output_dir}")

    if not (output_dir / TRANSFORMER_PATH).exists():
        print("⌛ Convert Transformer model")
        transformer_inputs = {
            "hidden_states": torch.ones([1, 48, 6, 44, 34]),
            "timestep": torch.zeros([1, 2244]).to(torch.float32),
            "encoder_hidden_states": torch.ones([1, 512, 4096]),
        }
        transformer.eval()
        __make_16bit_traceable(transformer)
        ts_decoder = TorchScriptPythonDecoder(transformer, example_input=transformer_inputs, trace_kwargs={"check_trace": False})
        with torch.no_grad():
            ov_model = ov.convert_model(ts_decoder, example_input=transformer_inputs)
        if compression_config is not None:
            ov_model = nncf.compress_weights(ov_model, **compression_config)
        ov.save_model(ov_model, output_dir / TRANSFORMER_PATH)
        del ov_model
        cleanup_torchscript_cache()
        print("✅ Transformer model successfully converted")

    del transformer
    gc.collect()


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for Wan pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


core = ov.Core()


class OVWanImageToVideoPipeline(DiffusionPipeline):
    def __init__(self, model_dir, device_map="CPU", ov_config=None):
        model_dir = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        scheduler = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")

        if isinstance(device_map, str):
            device_map = {
                "transformer": device_map,
                "text_encoder": device_map,
                "vae_encoder": device_map,
                "vae_decoder": device_map,
            }

        transformer_model = core.read_model(model_dir / TRANSFORMER_PATH)
        transformer = core.compile_model(transformer_model, device_map["transformer"], ov_config)
        text_encoder_model = core.read_model(model_dir / TEXT_ENCODER_PATH)
        text_encoder = core.compile_model(text_encoder_model, device_map["text_encoder"], ov_config)
        vae_encoder = core.compile_model(model_dir / VAE_ENCODER_PATH, device_map["vae_encoder"], ov_config)
        vae_decoder = core.compile_model(model_dir / VAE_DECODER_PATH, device_map["vae_decoder"], ov_config)

        super().__init__()

        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.z_dim = 48
        self.patch_size = [1, 2, 2]
        self.latents_mean = (
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.157,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.123,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.052,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
        )
        self.latents_std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.499,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.06,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]
        self.expand_timesteps = True

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = torch.from_numpy(self.text_encoder([text_input_ids, mask])[0])
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
    ):
        image = self.image_processor(images=image, return_tensors="pt")
        image_embeds = torch.from_numpy(self.image_encoder(image["pixel_values"])[0])
        # Get the second to last hidden state
        return image_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError("Cannot provide both `image` and `image_embeds`.")
        if image is None and image_embeds is None:
            raise ValueError("Must provide either `image` or `image_embeds`.")
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError("`image` must be a PIL Image or torch.Tensor")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 48,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]
        if self.expand_timesteps:
            video_condition = image

        elif last_image is None:
            video_condition = torch.cat([image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2)
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        if isinstance(generator, list):
            latent_condition = [torch.from_numpy(self.vae_encoder(video_condition)) for _ in generator]
            latent_condition = torch.cat(latent_condition, dim=0)
        else:
            latent_condition = torch.from_numpy(self.vae_encoder(video_condition)[0])
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.expand_timesteps:
            first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `torch.Tensor`):
                The input image to condition the video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation.
            height (`int`, defaults to `480`):
                The height in pixels of the generated video.
            width (`int`, defaults to `832`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in Classifier-Free Diffusion Guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A torch.Generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings.
            last_image (`torch.Tensor`, *optional*):
                Last frame image for continued generation.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a WanPipelineOutput instead of a plain tuple.
            max_sequence_length (`int`, defaults to `512`):
                Maximum sequence length for text encoder.

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, WanPipelineOutput is returned, otherwise a tuple is returned.
        """

        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.scheduler.timesteps
        image = self.video_processor.preprocess(image, height=height, width=width).to("cpu", dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to("cpu", dtype=torch.float32)
        # 6. Prepare latent variables
        latents, condition, first_frame_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents=self.z_dim,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            latents=latents,
            last_image=last_image,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Apply mask
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                # batch_size, seq_len
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                noise_pred = torch.from_numpy(self.transformer([latent_model_input, timestep, prompt_embeds])[0])
                if self.do_classifier_free_guidance:
                    noise_uncond = torch.from_numpy(self.transformer([latent_model_input, timestep, negative_prompt_embeds])[0])
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        if not output_type == "latent":
            latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = torch.from_numpy(self.vae_decoder(latents)[0])
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
