import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModelWithProjection
from diffusers import StableDiffusion3Pipeline
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import matplotlib.pyplot as plt
import numpy as np


def get_sd3_pipeline(model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder_3=None, tokenizer_3=None)
    return pipe


# This function takes in the models of a SD3 pipeline in the torch fx representation and returns an SD3 pipeline with wrapped models.
def init_pipeline(models_dict, configs_dict, model_id="stabilityai/stable-diffusion-3-medium-diffusers"):
    wrapped_models = {}

    def wrap_model(pipe_model, base_class, config):
        base_class = (base_class,) if not isinstance(base_class, tuple) else base_class

        class WrappedModel(*base_class):
            def __init__(self, model, config):
                cls_name = base_class[0].__name__
                if isinstance(config, dict):
                    super().__init__(**config)
                else:
                    super().__init__(config)
                if cls_name == "AutoencoderKL":
                    self.encoder = model.encoder
                    self.decoder = model.decoder
                else:
                    self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        class WrappedTransformer(*base_class):
            @register_to_config
            def __init__(
                self,
                model,
                sample_size,
                patch_size,
                in_channels,
                num_layers,
                attention_head_dim,
                num_attention_heads,
                joint_attention_dim,
                caption_projection_dim,
                pooled_projection_dim,
                out_channels,
                pos_embed_max_size,
                dual_attention_layers,
                qk_norm,
            ):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                del kwargs["joint_attention_kwargs"]
                del kwargs["return_dict"]
                return self.model(*args, **kwargs)

        if len(base_class) > 1:
            return WrappedTransformer(pipe_model, **config)
        return WrappedModel(pipe_model, config)

    wrapped_models["transformer"] = wrap_model(
        models_dict["transformer"],
        (
            ModelMixin,
            ConfigMixin,
        ),
        configs_dict["transformer"],
    )
    wrapped_models["vae"] = wrap_model(models_dict["vae"], AutoencoderKL, configs_dict["vae"])
    wrapped_models["text_encoder"] = wrap_model(models_dict["text_encoder"], CLIPTextModelWithProjection, configs_dict["text_encoder"])
    wrapped_models["text_encoder_2"] = wrap_model(models_dict["text_encoder_2"], CLIPTextModelWithProjection, configs_dict["text_encoder_2"])

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder_3=None, tokenizer_3=None, **wrapped_models)

    return pipe


def visualize_results(orig_img, optimized_img):
    """
    Helper function for results visualization copied from sd3_quantization_helper.py due to import conflicts.

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
