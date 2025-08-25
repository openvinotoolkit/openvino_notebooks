import os
import gc
import math
import torch
import types
import torchaudio
import torchvision.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from diffusers.utils.torch_utils import randn_tensor
from typing import Dict, Optional, List, Union, Type
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

import nncf
import openvino as ov
from openvino.tools.ovc import convert_model
from openvino_tokenizers import convert_tokenizer
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

from acestep.language_segmentation import LangSegment, language_filters
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.models.ace_step_transformer import Transformer2DModelOutput
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
from acestep.schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from acestep.schedulers.scheduling_flow_match_pingpong import FlowMatchPingPongScheduler
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)

torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER_MODEL_NAME = "openvino_tokenizer.xml"
TEXT_ENCODER_MODEL_NAME = "ov_text_encoder_model.xml"
DCAE_ENCODER_MODEL_NAME = "ov_dcae_encoder_model.xml"
DCAE_DECODER_MODEL_NAME = "ov_dcae_decoder_model.xml"
VOCODER_DECODE_MODEL_NAME = "ov_vocoder_decode_model.xml"
VOCODER_MEL_TRANSFORM_MODEL_NAME = "ov_vocoder_mel_transform_model.xml"
TRANSFORMER_DECODER_MODEL_NAME = "ov_transformer_decoder_model.xml"
TRANSFORMER_ENCODER_MODEL_NAME = "ov_transformer_encoder_model.xml"


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def ov_convert(
    model_dir_path: str,
    ov_model_name: str,
    inputs: Dict,
    orig_model: Type[torch.nn.Module],
    model_name: str,
    quantization_config: Dict = None,
    force_convertion: bool = False,
):
    try:
        ov_model_path = Path(model_dir_path, ov_model_name)
        if not ov_model_path.exists() or force_convertion:
            print(f"⌛ Convert {model_name} model")
            orig_model.eval()
            __make_16bit_traceable(orig_model)
            ov_model = convert_model(orig_model, example_input=inputs)
            if quantization_config is not None:
                print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
                ov_model = nncf.compress_weights(ov_model, **quantization_config)
                print("✅ Weights compression finished")
            ov.save_model(ov_model, ov_model_path)

            del ov_model
            cleanup_torchscript_cache()
            gc.collect()
            print(f"✅ {model_name} model converted")
    except Exception as e:
        print(f"❌{model_name} model is not converted. Error: {e}")


def convert_transformer_models(pipeline: ACEStepPipeline, model_dir: str = "ov_converted", orig_checkpoint_path: str = "", quantization_config: Dict = None):
    # Transformer Encoder model
    def encode_with_temperature_wrap(
        self,
        encoder_text_hidden_states: torch.Tensor = None,
        text_attention_mask: torch.LongTensor = None,
        speaker_embeds: torch.FloatTensor = None,
        lyric_token_idx: torch.LongTensor = None,
        lyric_mask: torch.LongTensor = None,
        tau: torch.FloatTensor = torch.Tensor([0.01]),
    ):
        handlers = []

        def hook(module, input, output):
            output[:] *= tau[0]
            return output

        l_min = 4
        l_max = 6
        for i in range(l_min, l_max):
            handler = self.lyric_encoder.encoders[i].self_attn.linear_q.register_forward_hook(hook)
            handlers.append(handler)

        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        for hook in handlers:
            hook.remove()

        return encoder_hidden_states, encoder_hidden_mask

    inputs = {
        "encoder_text_hidden_states": torch.randn(size=(1, 15, 768), dtype=torch.float),
        "text_attention_mask": torch.ones([1, 15], dtype=torch.int64),
        "speaker_embeds": torch.zeros(size=(1, 512), dtype=torch.float),
        "lyric_token_idx": torch.randint(10000, [1, 543], dtype=torch.int64),
        "lyric_mask": torch.ones([1, 543], dtype=torch.int64),
        "tau": torch.Tensor([0.01]),
    }
    transformer_encoder_model = pipeline.ace_step_transformer
    transformer_encoder_erg_model = pipeline.ace_step_transformer
    transformer_encoder_erg_model.forward = types.MethodType(encode_with_temperature_wrap, transformer_encoder_model)
    ov_convert(
        model_dir,
        TRANSFORMER_ENCODER_MODEL_NAME,
        inputs,
        transformer_encoder_erg_model,
        "Transformer Encoder with Entropy Rectifying Guidance",
        quantization_config=quantization_config,
    )

    # Transformer Decoder model
    def decode_with_temperature_wrap(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: torch.Tensor = None,
        # ssl_hidden_states: List[torch.Tensor] = None,
        output_length: int = 0,
        # block_controlnet_hidden_states: Union[List[torch.Tensor], torch.Tensor] = None,
        # controlnet_scale: Union[float, torch.Tensor] = 1.0,
        tau: torch.FloatTensor = torch.Tensor([0.01]),
    ):
        handlers = []

        def hook(module, input, output):
            output[:] *= tau[0]
            return output

        l_min = 5
        l_max = 10
        for i in range(l_min, l_max):
            handler = self.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
            handlers.append(handler)
            handler = self.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
            handlers.append(handler)

        sample = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            output_length=output_length,
            timestep=timestep,
        ).sample

        for hook in handlers:
            hook.remove()

        return sample

    inputs = {
        "hidden_states": torch.randn(size=(1, 8, 16, 151), dtype=torch.float),
        "attention_mask": torch.ones([1, 151], dtype=torch.int64),
        "encoder_hidden_states": torch.randn(size=(1, 559, 2560), dtype=torch.float),
        "encoder_hidden_mask": torch.ones([1, 559], dtype=torch.float),
        "output_length": torch.tensor(151),
        "timestep": torch.randn([1], dtype=torch.float),
        "tau": torch.Tensor([0.01]),
    }
    transformer_decoder_erg_model = pipeline.ace_step_transformer
    transformer_decoder_erg_model.forward = types.MethodType(decode_with_temperature_wrap, transformer_decoder_erg_model)
    ov_convert(
        model_dir,
        TRANSFORMER_DECODER_MODEL_NAME,
        inputs,
        transformer_decoder_erg_model,
        "Transformer Decoder with Entropy Rectifying Guidance",
        quantization_config=quantization_config,
    )


def convert_models(pipeline: ACEStepPipeline, model_dir: str = "ov_converted_new", orig_checkpoint_path: str = "", quantization_config: Dict = None):
    print(f"⌛ Conversion started. Be patient, it may takes some time.")

    if not pipeline.loaded or (orig_checkpoint_path and not Path(orig_checkpoint_path).exists()):
        print("⌛ Load Original model checkpoints")
        pipeline.load_checkpoint(orig_checkpoint_path)
        print("✅ Original model checkpoints successfully loaded")

    # Tokenizer
    ov_tokenizer_path = Path(model_dir, TOKENIZER_MODEL_NAME)
    if not ov_tokenizer_path.exists():
        print(f"⌛ Convert Tokenizer")
        if not ov_tokenizer_path.exists():
            ov_tokenizer = convert_tokenizer(pipeline.text_tokenizer, with_detokenizer=False)
            ov.save_model(ov_tokenizer, Path(model_dir, TOKENIZER_MODEL_NAME))
        print(f"✅ Tokenizer is converted")

    # Text Encoder Model
    inputs = {
        "input_ids": torch.randint(1000, size=(1, 15), dtype=torch.int64),
        "attention_mask": torch.ones([1, 15], dtype=torch.int64),
    }
    ov_convert(model_dir, TEXT_ENCODER_MODEL_NAME, inputs, pipeline.text_encoder_model, "UMT5 Encoder")

    # DCAE Encoder model
    inputs = {"hidden_states": torch.randn([1, 2, 128, 1208], dtype=torch.float)}
    ov_convert(model_dir, DCAE_ENCODER_MODEL_NAME, inputs, pipeline.music_dcae.dcae.encoder, "Sana's Deep Compression AutoEncoder")

    # DCAE Decoder model
    inputs = {"hidden_states": torch.randn([1, 8, 16, 151], dtype=torch.float)}
    ov_convert(model_dir, DCAE_DECODER_MODEL_NAME, inputs, pipeline.music_dcae.dcae.decoder, "Sana's Deep Compression AutoEncoder Decoder")

    # Vocoder Mel Transform model
    inputs = {"x": torch.randn([2, 618496], dtype=torch.float)}
    ov_convert(model_dir, VOCODER_MEL_TRANSFORM_MODEL_NAME, inputs, pipeline.music_dcae.vocoder.mel_transform, "Vocoder Mel Transform")

    # Vocoder Decoder model
    inputs = {"mel": torch.randn([1, 128, 856], dtype=torch.float)}
    ov_convert(model_dir, VOCODER_DECODE_MODEL_NAME, inputs, pipeline.music_dcae.vocoder, "Vocoder Decoder")

    # DiT
    convert_transformer_models(pipeline, model_dir, orig_checkpoint_path, quantization_config)


class MusicDCAEWrapper(MusicDCAE):
    def __init__(self, source_sample_rate=None):
        torch.nn.Module.__init__(self)
        self.dcae = None
        self.vocoder = None

        if source_sample_rate is None:
            source_sample_rate = 48000

        self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)

        self.transform = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091


class OVDCAECompiledModels(torch.nn.Module):
    def __init__(self, compiled_model):
        self.compiled_model = compiled_model

    def __call__(self, inputs):
        if not self.compiled_model:
            logger.error("OVDCAECompiledModels: compiled model is not defined")

        output = self.compiled_model({"hidden_states": inputs.to(dtype=torch.float32)})
        return torch.from_numpy(output[0])

    @classmethod
    def from_pretrained(cls, ov_model_path, device, ov_core):
        ov_dcae_model = ov_core.read_model(ov_model_path)
        compiled_model = ov_core.compile_model(ov_dcae_model, device)
        return cls(compiled_model)


class OVWrapperAutoencoderDC(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_pretrained(cls, ov_core, ov_models_path, device="CPU"):
        encoder = OVDCAECompiledModels.from_pretrained(Path(ov_models_path, DCAE_ENCODER_MODEL_NAME), device, ov_core)
        decoder = OVDCAECompiledModels.from_pretrained(Path(ov_models_path, DCAE_DECODER_MODEL_NAME), device, ov_core)
        return cls(encoder, decoder)


class OVWrapperADaMoSHiFiGANV1(torch.nn.Module):
    def __init__(self, encoder_compiled_model, mel_trnasform_compiled_model):
        super().__init__()
        self.decoder = encoder_compiled_model
        self.mel_trnasform = mel_trnasform_compiled_model

    @classmethod
    def from_pretrained(cls, ov_core, ov_models_path, device="CPU"):
        ov_vocoder_decoder_model = ov_core.read_model(Path(ov_models_path, VOCODER_DECODE_MODEL_NAME))
        decoder = ov_core.compile_model(ov_vocoder_decoder_model, device)
        ov_vocoder_mel_transform_model = ov_core.read_model(Path(ov_models_path, VOCODER_MEL_TRANSFORM_MODEL_NAME))
        mel_trnasform = ov_core.compile_model(ov_vocoder_mel_transform_model, device)
        return cls(decoder, mel_trnasform)

    def decode(self, inputs):
        output = self.decoder({"mel": inputs.to(dtype=torch.float32)})
        return torch.from_numpy(output[0])

    def mel_transform(self, inputs):
        output = self.mel_trnasform({"x": inputs.to(dtype=torch.float32)})
        return torch.from_numpy(output[0])

    def forward(self, inputs):
        return self.decode(inputs)


class OvWrapperACEStepTransformer2DModel(torch.nn.Module):
    def __init__(self, encoder_model, decoder_model):
        super().__init__()
        self.ov_lyric_encoder_compiled = encoder_model
        self.ov_decoder_compiled_model = decoder_model

    @classmethod
    def from_pretrained(cls, ov_core, ov_models_path, device="CPU"):
        ov_model_encoder = ov_core.read_model(Path(ov_models_path, TRANSFORMER_ENCODER_MODEL_NAME))
        compiled_model_encoder = ov_core.compile_model(ov_model_encoder, device)

        ov_model_decoder = ov_core.read_model(Path(ov_models_path, TRANSFORMER_DECODER_MODEL_NAME))
        compiled_model_decoder = ov_core.compile_model(ov_model_decoder, device)
        return cls(compiled_model_encoder, compiled_model_decoder)

    def encode_with_temperature(
        self,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        tau: Optional[torch.FloatTensor] = torch.Tensor([0.01]),
    ):
        output = None
        if self.ov_lyric_encoder_compiled:
            output = self.ov_lyric_encoder_compiled(
                {
                    "encoder_text_hidden_states": encoder_text_hidden_states,
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": speaker_embeds,
                    "lyric_token_idx": lyric_token_idx,
                    "lyric_mask": lyric_mask,
                    "tau": tau,
                }
            )
        return torch.from_numpy(output[0]), torch.from_numpy(output[1])

    def decode_with_temperature(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: Optional[torch.Tensor],
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        output_length: int = 0,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
        tau: Optional[torch.FloatTensor] = torch.Tensor([0.01]),
    ):
        output = None
        if self.ov_decoder_compiled_model:
            output = self.ov_decoder_compiled_model(
                {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_hidden_mask": encoder_hidden_mask,
                    "output_length": output_length,
                    "timestep": timestep,
                    "tau": tau,
                }
            )

        sample = torch.from_numpy(output[0]) if output is not None else None
        return sample

    def encode(
        self,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
    ):
        return self.encode_with_temperature(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
            tau=torch.Tensor([1]),
        )

    def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: Optional[torch.Tensor],
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        output_length: int = 0,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):
        sample = self.decode_with_temperature(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            ssl_hidden_states=ssl_hidden_states,
            output_length=output_length,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            controlnet_scale=controlnet_scale,
            return_dict=return_dict,
            tau=torch.Tensor([1]),
        )

        return Transformer2DModelOutput(sample, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.Tensor] = None,
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):
        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        output_length = hidden_states.shape[-1]

        output = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            ssl_hidden_states=ssl_hidden_states,
            output_length=output_length,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            controlnet_scale=controlnet_scale,
            return_dict=return_dict,
        )

        return output


class OVACEStepPipeline(ACEStepPipeline):
    def __init__(self):
        super().__init__(checkpoint_dir="", dtype="float32")
        self.core = ov.Core()

        self.dcae_decoder = None
        self.vocoder_encode = None
        self.vocoder_decoder = None
        self.transformer_encode = None
        self.transformer_encode_with_temperature = None
        self.transformer_decode = None
        self.transformer_decode_with_temperature = None

        self.ace_step_transformer_origin = None
        self.ace_step_transformer = None
        self.music_dcae = None
        self.text_tokenizer = None
        self.text_encoder_model = None

    def get_checkpoint_path(self, checkpoint_dir, repo):
        pass

    def load_checkpoint(self, checkpoint_dir=None, export_quantized_weights=False):
        pass

    def load_models(self, ov_models_path: str = None, device: str = "CPU"):
        self.loaded = True
        if ov_models_path and Path(ov_models_path).exists:
            ov_text_encoder_model = self.core.read_model(Path(ov_models_path, TEXT_ENCODER_MODEL_NAME))
            self.text_encoder_model = self.core.compile_model(ov_text_encoder_model, device)

            ov_text_tokenizer_path = self.core.read_model(Path(ov_models_path, TOKENIZER_MODEL_NAME))
            self.text_tokenizer = self.core.compile_model(ov_text_tokenizer_path, device)

            self.music_dcae = MusicDCAEWrapper()
            self.music_dcae.dcae = OVWrapperAutoencoderDC.from_pretrained(self.core, ov_models_path, device)
            self.music_dcae.vocoder = OVWrapperADaMoSHiFiGANV1.from_pretrained(self.core, ov_models_path, device)

            self.ace_step_transformer = OvWrapperACEStepTransformer2DModel.from_pretrained(self.core, ov_models_path, device)
        else:
            logger.error(f"Path is not exists: {ov_models_path}")

        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

    def load_quantized_checkpoint(self, checkpoint_dir=None):
        pass

    def get_text_embeddings(self, texts, text_max_length=256):
        inputs = self.text_tokenizer(texts)
        inputs = {"attention_mask": inputs["attention_mask"], "input_ids": inputs["input_ids"]}

        last_hidden_states = self.text_encoder_model(inputs)
        attention_mask = inputs["attention_mask"]
        return torch.from_numpy(last_hidden_states[0]), torch.from_numpy(attention_mask)

    def get_text_embeddings_null(self, texts, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        inputs = self.text_tokenizer(texts)
        inputs = {"attention_mask": inputs["attention_mask"], "input_ids": inputs["input_ids"]}
        last_hidden_states = self.text_encoder_model(inputs)
        return torch.from_numpy(last_hidden_states[0])

    def text2music_diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        repaint_start=0,
        repaint_end=0,
        src_latents=None,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_latents=None,
    ):
        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(cfg_type, guidance_scale, omega_scale))
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        do_double_condition_guidance = False
        if guidance_scale_text is not None and guidance_scale_text > 1.0 and guidance_scale_lyric is not None and guidance_scale_lyric > 1.0:
            do_double_condition_guidance = True
            logger.info(
                "do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(
                    do_double_condition_guidance,
                    guidance_scale_text,
                    guidance_scale_lyric,
                )
            )

        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )

        frame_length = int(duration * 44100 / 512 / 8)
        if src_latents is not None:
            frame_length = src_latents.shape[-1]

        if ref_latents is not None:
            frame_length = ref_latents.shape[-1]

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            scheduler.set_timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=infer_steps,
                device=self.device,
                timesteps=None,
            )
            new_timesteps = torch.zeros(len(oss_steps), dtype=self.dtype, device=self.device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx] - 1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=num_inference_steps,
                device=self.device,
                sigmas=sigmas,
            )
            logger.info(f"oss_steps: {oss_steps}, num_inference_steps: {num_inference_steps} after remapping to timesteps {timesteps}")
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=infer_steps,
                device=self.device,
                timesteps=None,
            )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=self.device,
            dtype=self.dtype,
        )

        is_repaint = False
        is_extend = False

        if add_retake_noise:
            n_min = int(infer_steps * (1 - retake_variance))
            retake_variance = torch.tensor(retake_variance * math.pi / 2).to(self.device).to(self.dtype)
            retake_latents = randn_tensor(
                shape=(bsz, 8, 16, frame_length),
                generator=retake_random_generators,
                device=self.device,
                dtype=self.dtype,
            )
            repaint_start_frame = int(repaint_start * 44100 / 512 / 8)
            repaint_end_frame = int(repaint_end * 44100 / 512 / 8)
            x0 = src_latents
            # retake
            is_repaint = repaint_end_frame - repaint_start_frame != frame_length

            is_extend = (repaint_start_frame < 0) or (repaint_end_frame > frame_length)
            if is_extend:
                is_repaint = True

            # TODO: train a mask aware repainting controlnet
            # to make sure mean = 0, std = 1
            if not is_repaint:
                target_latents = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
            elif not is_extend:
                # if repaint_end_frame
                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=self.device, dtype=self.dtype)
                repaint_mask[:, :, :, repaint_start_frame:repaint_end_frame] = 1.0
                repaint_noise = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
                repaint_noise = torch.where(repaint_mask == 1.0, repaint_noise, target_latents)
                zt_edit = x0.clone()
                z0 = repaint_noise
            elif is_extend:
                to_right_pad_gt_latents = None
                to_left_pad_gt_latents = None
                gt_latents = src_latents
                src_latents_length = gt_latents.shape[-1]
                max_infer_fame_length = int(240 * 44100 / 512 / 8)
                left_pad_frame_length = 0
                right_pad_frame_length = 0
                right_trim_length = 0
                left_trim_length = 0
                if repaint_start_frame < 0:
                    left_pad_frame_length = abs(repaint_start_frame)
                    frame_length = left_pad_frame_length + gt_latents.shape[-1]
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (left_pad_frame_length, 0), "constant", 0)
                    if frame_length > max_infer_fame_length:
                        right_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:, :, :, :max_infer_fame_length]
                        to_right_pad_gt_latents = extend_gt_latents[:, :, :, -right_trim_length:]
                        frame_length = max_infer_fame_length
                    repaint_start_frame = 0
                    gt_latents = extend_gt_latents

                if repaint_end_frame > src_latents_length:
                    right_pad_frame_length = repaint_end_frame - gt_latents.shape[-1]
                    frame_length = gt_latents.shape[-1] + right_pad_frame_length
                    extend_gt_latents = torch.nn.functional.pad(gt_latents, (0, right_pad_frame_length), "constant", 0)
                    if frame_length > max_infer_fame_length:
                        left_trim_length = frame_length - max_infer_fame_length
                        extend_gt_latents = extend_gt_latents[:, :, :, -max_infer_fame_length:]
                        to_left_pad_gt_latents = extend_gt_latents[:, :, :, :left_trim_length]
                        frame_length = max_infer_fame_length
                    repaint_end_frame = frame_length
                    gt_latents = extend_gt_latents

                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=self.device, dtype=self.dtype)
                if left_pad_frame_length > 0:
                    repaint_mask[:, :, :, :left_pad_frame_length] = 1.0
                if right_pad_frame_length > 0:
                    repaint_mask[:, :, :, -right_pad_frame_length:] = 1.0
                x0 = gt_latents
                padd_list = []
                if left_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, :left_pad_frame_length])
                padd_list.append(
                    target_latents[
                        :,
                        :,
                        :,
                        left_trim_length : target_latents.shape[-1] - right_trim_length,
                    ]
                )
                if right_pad_frame_length > 0:
                    padd_list.append(retake_latents[:, :, :, -right_pad_frame_length:])
                target_latents = torch.cat(padd_list, dim=-1)
                assert target_latents.shape[-1] == x0.shape[-1], f"{target_latents.shape=} {x0.shape=}"
                zt_edit = x0.clone()
                z0 = target_latents

        if audio2audio_enable and ref_latents is not None:
            logger.info(f"audio2audio_enable: {audio2audio_enable}, ref_latents: {ref_latents.shape}")
            target_latents, timesteps, scheduler, num_inference_steps = self.add_latents_noise(
                gt_latents=ref_latents,
                sigma_max=(1 - ref_audio_strength),
                noise=target_latents,
                scheduler_type=scheduler_type,
                infer_steps=infer_steps,
            )

        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=self.dtype)

        # guidance interval
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info(f"start_idx: {start_idx}, end_idx: {end_idx}, num_inference_steps: {num_inference_steps}")

        momentum_buffer = MomentumBuffer()

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode_with_temperature(
                encoder_text_hidden_states=(
                    encoder_text_hidden_states_null if encoder_text_hidden_states_null is not None else torch.zeros_like(encoder_text_hidden_states)
                ),
                text_attention_mask=text_attention_mask,
                speaker_embeds=torch.zeros_like(speaker_embds),
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
            )
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )

        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            # P(null_speaker, text, lyric_weaker)
            if use_erg_lyric:
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode_with_temperature(
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embeds=torch.zeros_like(speaker_embds),
                    lyric_token_idx=lyric_token_ids,
                    lyric_mask=lyric_mask,
                )
            # P(null_speaker, text, no_lyric)
            else:
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids),
                    lyric_mask,
                )

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            if is_repaint:
                if i < n_min:
                    continue
                elif i == n_min:
                    t_i = t / 1000
                    zt_src = (1 - t_i) * x0 + (t_i) * z0
                    target_latents = zt_edit + zt_src - x0
                    logger.info(f"repaint start from {n_min} add {t_i} level of noise")

            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    progress = (i - start_idx) / (end_idx - start_idx - 1)  # 归一化到[0,1]
                    current_guidance_scale = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)
                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred_with_only_text_cond = None
                if do_double_condition_guidance and encoder_hidden_states_no_lyric is not None:
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = self.ace_step_transformer.decode_with_temperature(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        attention_mask=attention_mask,
                    )
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if do_double_condition_guidance and noise_pred_with_only_text_cond is not None:
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                    )

                elif cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=current_guidance_scale,
                    )
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(
                        noise_pred_with_cond=noise_pred_with_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        i=i,
                        zero_steps=zero_steps,
                        use_zero_init=use_zero_init,
                    )
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
                    timestep=timestep,
                ).sample

            if is_repaint and i >= n_min:
                t_i = t / 1000
                if i + 1 < len(timesteps):
                    t_im1 = (timesteps[i + 1]) / 1000
                else:
                    t_im1 = torch.zeros_like(t_i).to(self.device)
                target_latents = target_latents.to(torch.float32)
                prev_sample = target_latents + (t_im1 - t_i) * noise_pred
                prev_sample = prev_sample.to(self.dtype)
                target_latents = prev_sample
                zt_src = (1 - t_im1) * x0 + (t_im1) * z0
                target_latents = torch.where(repaint_mask == 1.0, target_latents, zt_src)
            else:
                target_latents = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=target_latents,
                    return_dict=False,
                    omega=omega_scale,
                    generator=random_generators[0],
                )[0]

        if is_extend:
            if to_right_pad_gt_latents is not None:
                target_latents = torch.cat([target_latents, to_right_pad_gt_latents], dim=-1)
            if to_left_pad_gt_latents is not None:
                target_latents = torch.cat([to_right_pad_gt_latents, target_latents], dim=0)
        return target_latents

    def load_lora(self, model_with_lora_path, device="CPU"):
        if model_with_lora_path == "none":
            if self.ace_step_transformer_origin:
                self.ace_step_transformer = self.ace_step_transformer_origin
        else:
            self.ace_step_transformer_origin = self.ace_step_transformer
            self.update_transformer_model(model_with_lora_path, device)

    def update_transformer_model(self, new_transformer_path, device="CPU"):
        self.ace_step_transformer = OvWrapperACEStepTransformer2DModel.from_pretrained(self.core, new_transformer_path, device)
