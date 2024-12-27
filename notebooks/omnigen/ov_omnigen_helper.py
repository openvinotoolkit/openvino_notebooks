from pathlib import Path
from types import MethodType
from typing import Union, List, Optional, Tuple, Dict, Any
from OmniGen import OmniGenPipeline, OmniGenProcessor
from OmniGen.scheduler import OmniGenCache, OmniGenScheduler
from OmniGen.model import get_2d_sincos_pos_embed
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig
from PIL import Image
import openvino as ov
import numpy as np
import nncf
import torch
import gc
import math
import json
from tqdm.auto import tqdm
from copy import deepcopy


class OVOmniGenScheduler(OmniGenScheduler):
    def __call__(self, z, func, model_kwargs, model_config):
        num_tokens_for_img = z.size(-1) * z.size(-2) // 4
        if isinstance(model_kwargs["input_ids"], list):
            cache = [OmniGenCacheWrapper(num_tokens_for_img, False, model_config=model_config) for _ in range(len(model_kwargs["input_ids"]))]
        else:
            cache = OmniGenCache(num_tokens_for_img, False, model_config=model_config)
        for i in tqdm(range(self.num_steps)):
            timesteps = torch.zeros(size=(len(z),)).to(z.device) + self.sigma[i]
            pred, cache = func(z, timesteps, past_key_values=cache, **model_kwargs)
            sigma_next = self.sigma[i + 1]
            sigma = self.sigma[i]
            z = z + (sigma_next - sigma) * pred
            if i == 0:
                num_tokens_for_img = z.size(-1) * z.size(-2) // 4
                if isinstance(cache, list):
                    model_kwargs["input_ids"] = [None] * len(cache)
                else:
                    model_kwargs["input_ids"] = None

                model_kwargs["position_ids"] = self.crop_position_ids_for_cache(model_kwargs["position_ids"], num_tokens_for_img)
                model_kwargs["attention_mask"] = self.crop_attention_mask_for_cache(model_kwargs["attention_mask"], num_tokens_for_img)

        del cache
        gc.collect()
        return z


class OmniGenCacheWrapper(DynamicCache):
    def __init__(self, num_tokens_for_img: int, key_states=None, value_states=None, model_config=None) -> None:
        super().__init__()
        self.num_tokens_for_img = num_tokens_for_img
        if key_states is not None:
            self.key_cache = key_states or []
        if value_states is not None:
            self.value_cache = value_states or []
        self.model_config = model_config

    def to_openvino_inputs(self):
        pkv_inputs = {}
        if not self.key_cache:
            hidden_size = self.model_config.hidden_size
            num_key_value_heads = self.model_config.num_key_value_heads
            num_attention_heads = self.model_config.num_attention_heads

            pkv_tensor = np.zeros([1, num_key_value_heads, 0, hidden_size // num_attention_heads])

            for i in range(self.model_config.num_hidden_layers):
                pkv_inputs[f"past_key_values.{i}.key"] = pkv_tensor
                pkv_inputs[f"past_key_values.{i}.value"] = pkv_tensor
        else:
            for idx, key in enumerate(self.key_cache):
                pkv_inputs[f"past_key_values.{idx}.key"] = key

            for idx, value in enumerate(self.value_cache):
                pkv_inputs[f"past_key_values.{idx}.value"] = value
        return pkv_inputs

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            # only cache the states for condition tokens
            key_states = key_states[..., : -(self.num_tokens_for_img + 1), :]
            value_states = value_states[..., : -(self.num_tokens_for_img + 1), :]

            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            # only cache the states for condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = torch.cat([key_tensor, key_states], dim=-2)
            v = torch.cat([value_tensor, value_states], dim=-2)
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
            return k, v

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")


# model_id = "Shitao/OmniGen-v1"
# model_path = Path("omnigen_ov")


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_omingen_model(model_id, model_path, quant_config=None):
    vae_encoder_path = model_path / "vae/vae_encoder.xml"
    vae_decoder_path = model_path / "vae/vae_decoder.xml"
    llm_embed_tokens_path = model_path / "llm/input_embed.xml"
    llm_embed_t_path = model_path / "llm/timestep_embed.xml"
    llm_final_layer_path = model_path / "llm/final_layer.xml"
    llm_path = model_path / "llm/language_model.xml"
    llm_x_embedder_path = model_path / "llm/x_embedder.xml"
    llm_input_x_embedder_path = model_path / "llm/input_x_embedder.xml"
    llm_time_token_path = model_path / "llm/time_token.xml"
    required_conversion = not all(
        [
            vae_encoder_path.exists(),
            vae_decoder_path.exists(),
            llm_embed_t_path.exists(),
            llm_path.exists(),
            llm_embed_tokens_path.exists(),
            llm_final_layer_path.exists(),
            llm_input_x_embedder_path.exists(),
            llm_x_embedder_path.exists(),
            llm_time_token_path.exists(),
        ]
    )
    if not required_conversion:
        print(f"Model already converted and can be found in {model_path}")
        return
    pipe = OmniGenPipeline.from_pretrained(model_id)
    pipe.processor.text_tokenizer.save_pretrained(model_path)
    need_to_convert_llm = not all(
        [
            llm_path.exists(),
            llm_embed_tokens_path.exists(),
            llm_embed_t_path.exists(),
            llm_final_layer_path.exists(),
            llm_input_x_embedder_path.exists(),
            llm_x_embedder_path.exists(),
            llm_time_token_path.exists(),
        ]
    )
    if need_to_convert_llm:
        print("LLM conversion started...")
        pipe.model.llm.config.save_pretrained(model_path / "llm")
        if not llm_embed_tokens_path.exists():
            ov_model = ov.convert_model(pipe.model.llm.embed_tokens, example_input=torch.ones([1, 1], dtype=torch.long))
            ov.save_model(ov_model, llm_embed_tokens_path)
            del ov_model
            gc.collect()
        if not llm_path.exists():
            hidden_size = pipe.model.llm.config.hidden_size
            num_key_value_heads = pipe.model.llm.config.num_key_value_heads
            num_attention_heads = pipe.model.llm.config.num_attention_heads
            num_hidden_layers = pipe.model.llm.config.num_hidden_layers
            pkv_seq_len = 36
            current_seq_len = 257
            input_embeds = torch.randn([1, current_seq_len, hidden_size])
            attention_mask = torch.ones([1, current_seq_len, current_seq_len + pkv_seq_len], dtype=torch.long)
            position_ids = torch.range(pkv_seq_len, current_seq_len + pkv_seq_len - 1, dtype=torch.long).unsqueeze(0)
            example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids}

            pkv_shape = [1, num_key_value_heads, pkv_seq_len, hidden_size // num_attention_heads]
            input_names = ["attention_mask", "position_ids", "inputs_embeds"]
            output_names = ["last_hidden_state"]
            past_key_values = []

            def forward_wrap(
                self,
                attention_mask,
                position_ids=None,
                inputs_embeds=None,
                past_key_values=None,
            ):
                if past_key_values is not None:
                    key_states = [kv[0] for kv in past_key_values]
                    value_states = [kv[1] for kv in past_key_values]
                    past_key_values = OmniGenCacheWrapper(256, key_states=key_states, value_states=value_states)
                result = self._orig_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
                result.past_key_values = result.past_key_values.to_legacy_cache()
                return tuple(result.values())

            @torch.jit.script
            def select_ext_factor(seq_len: torch.Tensor, max_pos_embeddings: torch.Tensor, short_factor: torch.Tensor, long_factor: torch.Tensor):
                if seq_len > max_pos_embeddings:
                    return long_factor
                return short_factor

            def rope_fwd(self, x, position_ids, seq_len=None):
                seq_len = torch.tensor(seq_len) or torch.max(position_ids) + 1
                ext_factors = select_ext_factor(
                    seq_len,
                    torch.tensor(self.original_max_position_embeddings),
                    torch.tensor(self.short_factor, dtype=torch.float32, device=x.device),
                    torch.tensor(self.long_factor, dtype=torch.float32, device=x.device),
                )

                inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
                inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

                inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
                position_ids_expanded = position_ids[:, None, :].float()

                # Force float32 since bfloat16 loses precision on long contexts
                # See https://github.com/huggingface/transformers/pull/29285
                device_type = x.device.type
                device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                    emb = torch.cat((freqs, freqs), dim=-1)

                    scale = self.max_position_embeddings / self.original_max_position_embeddings
                    if scale <= 1.0:
                        scaling_factor = 1.0
                    else:
                        scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
                    cos = emb.cos() * scaling_factor
                    sin = emb.sin() * scaling_factor
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            pipe.model.llm._orig_forward = pipe.model.llm.forward
            pipe.model.llm.forward = MethodType(forward_wrap, pipe.model.llm)
            for layer in pipe.model.llm.layers:
                layer.self_attn.rotary_emb.forward = MethodType(rope_fwd, layer.self_attn.rotary_emb)
            for i in range(num_hidden_layers):
                past_key_values.append((torch.randn(pkv_shape), torch.randn(pkv_shape)))
                input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])
            example_input["past_key_values"] = past_key_values

            ov_model = ov.convert_model(pipe.model.llm, example_input=example_input)
            for input_name, input_tensor in zip(input_names, ov_model.inputs):
                input_tensor.set_names({input_name})
            for out_name, out_tensor in zip(output_names, ov_model.outputs):
                out_tensor.set_names({out_name})
            llm_saving_path = llm_path if quant_config is None else llm_path.parent / ("tmp_" + llm_path.name)
            ov.save_model(ov_model, llm_saving_path)
            del ov_model
            cleanup_torchscript_cache()
            del pipe.model.llm
            pipe.model.llm = None
            gc.collect()
            if quant_config is not None:
                ov_model = ov.Core().read_model(llm_saving_path)
                ov_compressed_model = nncf.compress_weights(ov_model, **quant_config)
                ov.save_model(ov_compressed_model, llm_path)
                del ov_compressed_model
                del ov_model
                gc.collect()
                llm_saving_path.unlink()
                (llm_saving_path.with_suffix(".bin")).unlink()
        del pipe.model.llm
        gc.collect()
        if not llm_embed_t_path.exists():
            ov_model = ov.convert_model(pipe.model.t_embedder, example_input=torch.tensor([0.5]))
            ov.save_model(ov_model, llm_embed_t_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()

        if not llm_final_layer_path.exists():
            ov_model = ov.convert_model(pipe.model.final_layer, example_input=(torch.randn(1, 256, 3072), torch.randn(1, 3072)))
            ov.save_model(ov_model, llm_final_layer_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()

        if not llm_x_embedder_path.exists():
            ov_model = ov.convert_model(pipe.model.x_embedder, example_input=torch.randn([1, 4, 32, 32]))
            ov.save_model(ov_model, llm_x_embedder_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()
        if not llm_input_x_embedder_path.exists():
            ov_model = ov.convert_model(pipe.model.input_x_embedder, example_input=torch.randn([1, 4, 32, 32]))
            ov.save_model(ov_model, llm_input_x_embedder_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()
        if not llm_time_token_path.exists():
            ov_model = ov.convert_model(pipe.model.time_token, example_input=torch.tensor([0.5]))
            ov.save_model(ov_model, llm_time_token_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()

        del pipe.model
        gc.collect()

        print("LLM conversion finished")

    if not vae_encoder_path.exists() or not vae_decoder_path.exists():
        print("VAE conversion started...")
        pipe.vae.save_config(model_path / "vae")
        if not vae_encoder_path.exists():
            pipe.vae.forward = pipe.vae._encode
            ov_model = ov.convert_model(pipe.vae, example_input=torch.randn((1, 3, 256, 256)))
            ov.save_model(ov_model, vae_encoder_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()
        if not vae_decoder_path.exists():
            pipe.vae.forward = pipe.vae.decode
            ov_model = ov.convert_model(pipe.vae, example_input=torch.randn((1, 4, 32, 32)))
            ov.save_model(ov_model, vae_decoder_path)
            del ov_model
            cleanup_torchscript_cache()
            gc.collect()


# convert_model(model_id, model_path)

core = ov.Core()


class OVVaeModel:
    def __init__(self, model_dir, device, ov_config=None) -> None:
        self.vae_encoder = core.compile_model(model_dir / "vae_encoder.xml", device.upper(), ov_config)
        self.vae_decoder = core.compile_model(model_dir / "vae_decoder.xml", device.upper(), ov_config)
        json_file = model_dir / "config.json"
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        self.config = json.loads(text)

    def encode(self, image):
        z = self.vae_encoder(image)[0]
        return DiagonalGaussianDistribution(torch.from_numpy(z))

    def decode(self, sample):
        sample = self.vae_decoder(sample)[0]
        return ModelOutput(sample=torch.from_numpy(sample))


class OvModelForCausalLMWithEmb:
    def __init__(self, model_dir, device="CPU", config=None, ov_config=None, compile=True) -> None:
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir) if config is None else config
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "language_model.xml")
        self.token_emb = core.read_model(model_dir / "input_embed.xml")
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config
        self.next_beam_idx = None
        self._past_length = None
        self.input_names = [input_t.get_any_name() for input_t in self.model.inputs]
        self.main_input_name = "input_ids"
        if compile:
            self.compile()

    def compile(self):
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_token_emb()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = core.compile_model(self.token_emb, self._device, self.ov_config)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        inputs = {}
        past = past_key_values.to_openvino_inputs()
        inputs.update(past)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            inputs["position_ids"] = position_ids

        return inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        self.compile()

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        last_hidden_state = self.request.get_tensor("last_hidden_state").data
        if past_key_values._seen_tokens == 0:
            for i in range(self.config.num_hidden_layers):
                key_name = f"present.{i}.key"
                value_name = f"present.{i}.value"
                past_key_values.update(deepcopy(self.request.get_tensor(key_name).data), deepcopy(self.request.get_tensor(value_name).data), i)

        return ModelOutput(last_hidden_state=last_hidden_state, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVOmniGenModel:
    def __init__(
        self,
        model_dir,
        device,
        ov_config=None,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ) -> None:
        self.x_embedder = core.compile_model(model_dir / "x_embedder.xml", device.upper(), ov_config)
        self.t_embedder = core.compile_model(model_dir / "timestep_embed.xml", device.upper(), ov_config)
        self.input_x_embedder = core.compile_model(model_dir / "input_x_embedder.xml", device.upper(), ov_config)
        self.final_layer = core.compile_model(model_dir / "final_layer.xml", device.upper(), ov_config)
        self.llm = OvModelForCausalLMWithEmb(model_dir, device, ov_config=ov_config)
        self.time_token = core.compile_model(model_dir / "time_token.xml", device.upper(), ov_config)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.pe_interpolation = pe_interpolation
        self.pos_embed = get_2d_sincos_pos_embed(self.llm.config.hidden_size, pos_embed_max_size, interpolation_scale=self.pe_interpolation, base_size=64)

    @torch.no_grad()
    def forward_with_cfg(
        self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values
    ):
        model_out, past_key_values = self.forward(
            x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, past_key_values=past_key_values
        )
        if use_img_cfg:
            cond, uncond, img_cond = torch.split(model_out, len(model_out) // 3, dim=0)
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        else:
            cond, uncond = torch.split(model_out, len(model_out) // 2, dim=0)
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]

        return torch.cat(model_out, dim=0), past_key_values

    @torch.no_grad()
    def forward_with_separate_cfg(
        self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values
    ):
        if past_key_values is None:
            past_key_values = [None] * len(attention_mask)

        x = torch.split(x, len(x) // len(attention_mask), dim=0)
        timestep = timestep.to(x[0].dtype)
        timestep = torch.split(timestep, len(timestep) // len(input_ids), dim=0)

        model_out, pask_key_values = [], []
        for i in range(len(input_ids)):
            temp_out, temp_pask_key_values = self.forward(
                x[i],
                timestep[i],
                input_ids[i],
                input_img_latents[i],
                input_image_sizes[i],
                attention_mask[i],
                position_ids[i],
                past_key_values=past_key_values[i],
            )
            model_out.append(temp_out)
            pask_key_values.append(temp_pask_key_values)

        if len(model_out) == 3:
            cond, uncond, img_cond = model_out
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        elif len(model_out) == 2:
            cond, uncond = model_out
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        else:
            return model_out[0]

        return torch.cat(model_out, dim=0), pask_key_values

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels

        x = x.reshape(shape=(x.shape[0], h // self.patch_size, w // self.patch_size, self.patch_size, self.patch_size, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}.")
        if width > self.pos_embed_max_size:
            raise ValueError(f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}.")

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        # print(top, top + height, left, left + width, spatial_pos_embed.size())
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images: bool = False):
        if isinstance(latents, list):
            return_list = False
            if padding_latent is None:
                padding_latent = [None] * len(latents)
                return_list = True
            patched_latents, num_tokens, shapes = [], [], []
            for latent, padding in zip(latents, padding_latent):
                height, width = latent.shape[-2:]
                if is_input_images:
                    latent = torch.from_numpy(self.input_x_embedder(latent)[0])
                else:
                    latent = torch.from_numpy(self.x_embedder(latent)[0])
                pos_embed = self.cropped_pos_embed(height, width)
                latent = latent + pos_embed
                if padding is not None:
                    latent = torch.cat([latent, padding], dim=-2)
                patched_latents.append(latent)

                num_tokens.append(pos_embed.shape[1])
                shapes.append([height, width])
            if not return_list:
                latents = torch.cat(patched_latents, dim=0)
            else:
                latents = patched_latents
        else:
            height, width = latents.shape[-2:]
            if is_input_images:
                latents = torch.from_numpy(self.input_x_embedder(latents)[0])
            else:
                latents = torch.from_numpy(self.x_embedder(latents)[0])
            pos_embed = self.cropped_pos_embed(height, width)
            latents = latents + pos_embed
            num_tokens = latents.shape[1]
            shapes = [height, width]
        return latents, num_tokens, shapes

    def forward(
        self,
        x,
        timestep,
        input_ids,
        input_img_latents,
        input_image_sizes,
        attention_mask,
        position_ids,
        padding_latent=None,
        past_key_values=None,
        return_past_key_values=True,
    ):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        time_token = torch.from_numpy(self.time_token(timestep)[0]).unsqueeze(1)

        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
        if input_ids is not None:
            condition_embeds = torch.from_numpy(self.llm.embed_tokens(input_ids))
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds[b_inx, start_inx:end_inx] = input_latents[input_img_inx]
                    input_img_inx += 1
            if input_img_latents is not None:
                assert input_img_inx == len(input_latents)

            input_emb = torch.cat([condition_embeds, time_token, x], dim=1)
        else:
            input_emb = torch.cat([time_token, x], dim=1)
        output = self.llm(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
        output, past_key_values = output.last_hidden_state, output.past_key_values
        if input_is_list:
            image_embedding = output[:, -max(num_tokens) :]
            time_emb = self.t_embedder(timestep)[0]
            x = self.final_layer([image_embedding, time_emb])[0]
            latents = []
            for i in range(x.shape[0]):
                latent = x[i : i + 1, : num_tokens[i]]
                latent = self.unpatchify(torch.from_numpy(latent), shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep)[0]
            x = self.final_layer([image_embedding, time_emb])[0]
            latents = self.unpatchify(torch.from_numpy(x), shapes[0], shapes[1])

        if return_past_key_values:
            return latents, past_key_values
        return latents


class OVOmniGenPipeline:
    def __init__(self, model_dir, device, ov_config=None) -> None:
        self.processor = OmniGenProcessor.from_pretrained(model_dir)
        self.model = OVOmniGenModel(model_dir / "llm", device, ov_config)
        self.vae = OVVaeModel(model_dir / "vae", device, ov_config)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = False,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.float32,
        seed: int = None,
        output_type: str = "pil",
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):return
                The prompt or prompts to guide the image generation.
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800).
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800).
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output.
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            output_type (`str`, *optional*, defaults to "pil"):
                The type of the output image, which can be "pt" or "pil"
        Examples:

        Returns:
            A list with the generated images.
        """
        # check inputs:
        if use_input_image_size_as_output:
            assert (
                isinstance(prompt, str) and len(input_images) == 1
            ), "if you want to make sure the output image have the same size as the input image, please only input one image instead of multiple input images"
        else:
            assert height % 16 == 0 and width % 16 == 0, "The height and width must be a multiple of 16."
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None

        # set model and processor
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size)

        input_data = self.processor(
            prompt,
            input_images,
            height=height,
            width=width,
            use_img_cfg=use_img_guidance,
            separate_cfg_input=separate_cfg_infer,
            use_input_image_size_as_output=use_input_image_size_as_output,
        )
        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data["input_pixel_values"][0][0].shape[-2:]
            else:
                height, width = input_data["input_pixel_values"][0].shape[-2:]
        latent_size_h, latent_size_w = height // 8, width // 8

        if seed is not None:
            generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=torch.device("cpu"), generator=generator)
        latents = torch.cat([latents] * (1 + num_cfg), 0).to(dtype)
        input_img_latents = []
        if separate_cfg_infer:
            for temp_pixel_values in input_data["input_pixel_values"]:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(img, dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data["input_pixel_values"]:
                img = self.vae_encode(img, dtype)
                input_img_latents.append(img)
        model_kwargs = dict(
            input_ids=input_data["input_ids"],
            input_img_latents=input_img_latents,
            input_image_sizes=input_data["input_image_sizes"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data["position_ids"],
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
        )

        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        scheduler = OVOmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, model_config=self.model.llm.config)
        samples = samples.chunk((1 + num_cfg), dim=0)[0]
        samples = samples.to(torch.float32)
        if self.vae.config.get("shift_factor") is not None:
            samples = samples / self.vae.config["scaling_factor"] + self.vae.config["shift_factor"]
        else:
            samples = samples / self.vae.config["scaling_factor"]
        samples = self.vae.decode(samples).sample

        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        if output_type == "pt":
            output_images = samples
        else:
            output_samples = (samples * 255).to("cpu", dtype=torch.uint8)
            output_samples = output_samples.permute(0, 2, 3, 1).numpy()
            output_images = []
            for i, sample in enumerate(output_samples):
                output_images.append(Image.fromarray(sample))

        return output_images

    def vae_encode(self, x, dtype):
        if self.vae.config.get("shift_factor") is not None:
            x = self.vae.encode(x).sample()
            x = (x - self.vae.config["shift_factor"]) * self.vae.config["scaling_factor"]
        else:
            x = self.vae.encode(x).sample().mul_(self.vae.config["scaling_factor"])
        x = x.to(dtype)
        return x


# pipe = OVOmniGenPipeline(model_path, "CPU")

# print("T2I")

# images = pipe(
#     prompt="A curly-haired man in a red shirt is drinking tea.",
#     height=256,
#     width=256,
#     guidance_scale=2.5,
#     seed=0,
#     offload_kv_cache=False,
#     max_input_image_size=256,
#     num_inference_steps=20
# )
# images[0].save("example_t2i.png")

# print("MultiModal 2 I")

# images = pipe(
#     prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
#     input_images=["./imgs/test_cases/two_man.jpg"],
#     height=256,
#     width=256,
#     guidance_scale=2.5,
#     img_guidance_scale=1.6,
#     seed=0,
#     offload_kv_cache=False,
#     max_input_image_size=256,
#     num_inference_steps=20
# )
# images[0].save("example_ti2i.png")
