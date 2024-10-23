import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModelWithProjection
from diffusers import StableDiffusion3Pipeline
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import StableDiffusion3Pipeline

def get_sd3_pipeline(model_id='stabilityai/stable-diffusion-3-medium-diffusers'):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, text_encoder_3=None, tokenizer_3=None)
    return pipe

# This function takes in the models of a SD3 pipeline in the torch fx representation and returns an SD3 pipeline with wrapped models.
def init_pipeline(models_dict, configs_dict):
    wrapped_models = {}
    def wrap_model(pipe_model, base_class, config):
        base_class = (base_class,) if not isinstance(base_class, tuple) else base_class
        class WrappedModel(*base_class):
            def __init__(self, model, config):
                cls_name = base_class[0].__name__
                if(isinstance(config, dict)):
                    super().__init__(**config)
                else:
                    super().__init__(config)
                if(cls_name=='AutoencoderKL'):
                    self.encoder = model.encoder
                    self.decoder = model.decoder
                else:
                    self.model = model
            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)
        class WrappedTransformer(*base_class):
            ConfigMixin.ignore_for_config = []
            @register_to_config
            def __init__(self, model, 
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
                        pos_embed_max_size
                        ):
                super().__init__()
                self.model = model
            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)   
        if len(base_class) > 1: 
            return WrappedTransformer(pipe_model, **config)
        return WrappedModel(pipe_model, config)

    wrapped_models['transformer'] = wrap_model(models_dict['transformer'], (ModelMixin, ConfigMixin,), configs_dict['transformer'])
    wrapped_models['vae'] = wrap_model(models_dict['vae'], AutoencoderKL, configs_dict['vae'])
    wrapped_models['text_encoder'] = wrap_model(models_dict['text_encoder'], CLIPTextModelWithProjection, configs_dict['text_encoder'])
    wrapped_models['text_encoder_2'] = wrap_model(models_dict['text_encoder_2'], CLIPTextModelWithProjection, configs_dict['text_encoder_2'])
    
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None, **wrapped_models)

    return pipe