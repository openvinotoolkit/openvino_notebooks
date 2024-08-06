import gc
from pathlib import Path
from functools import partial
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from peft import PeftModel
import openvino as ov


MODEL_DIR = Path("stable-diffusion-3")

TRANSFORMER_PATH = MODEL_DIR / "transformer.xml"
VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder.xml"
TEXT_ENCODER_2_PATH = MODEL_DIR / "text_encoder_2.xml"
TEXT_ENCODER_3_PATH = MODEL_DIR / "text_encoder_3.xml"


def get_pipeline_components(use_flash_lora, load_t5):
    pipe_kwargs = {}
    if use_flash_lora:
        # Load LoRA
        transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
        )
        transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3")
        pipe_kwargs["transformer"] = transformer
    if not load_t5:
        pipe_kwargs.update({"text_encoder_3": None, "tokenizer_3": None})
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", **pipe_kwargs)
    pipe.tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    pipe.tokenizer_2.save_pretrained(MODEL_DIR / "tokenizer_2")
    if load_t5:
        pipe.tokenizer_3.save_pretrained(MODEL_DIR / "tokenizer_3")
    pipe.scheduler.save_pretrained(MODEL_DIR / "scheduler")
    transformer, vae, text_encoder, text_encoder_2, text_encoder_3 = None, None, None, None, None
    if not TRANSFORMER_PATH.exists():
        transformer = pipe.transformer
        transformer.eval()
    if not VAE_DECODER_PATH.exists():
        vae = pipe.vae
        vae.eval()
    if not TEXT_ENCODER_PATH.exists():
        text_encoder = pipe.text_encoder
        text_encoder.eval()
    if not TEXT_ENCODER_2_PATH.exists():
        text_encoder_2 = pipe.text_encoder_2
        text_encoder_2.eval()
    if not TEXT_ENCODER_3_PATH.exists() and load_t5:
        text_encoder_3 = pipe.text_encoder_3
        text_encoder_3.eval()
    return transformer, vae, text_encoder, text_encoder_2, text_encoder_3

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_transformer(transformer):
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, return_dict=False):
            return self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=return_dict,
            )

    if isinstance(transformer, PeftModel):
        transformer = TransformerWrapper(transformer)
    transformer.forward = partial(transformer.forward, return_dict=False)

    with torch.no_grad():
        ov_model = ov.convert_model(
            transformer,
            example_input={
                "hidden_states": torch.zeros((2, 16, 64, 64)),
                "timestep": torch.tensor([1, 1]),
                "encoder_hidden_states": torch.ones([2, 154, 4096]),
                "pooled_projections": torch.ones([2, 2048]),
            },
        )
    ov.save_model(ov_model, TRANSFORMER_PATH)
    del ov_model
    cleanup_torchscript_cache()



def convert_sd3(load_t5, use_flash_lora):
    conversion_statuses = [TRANSFORMER_PATH.exists(), VAE_DECODER_PATH.exists(), TEXT_ENCODER_PATH.exists(), TEXT_ENCODER_2_PATH.exists()]

    if load_t5:
        conversion_statuses.append(TEXT_ENCODER_3_PATH.exists())

    requires_conversion = not all(conversion_statuses)

    transformer, vae, text_encoder, text_encoder_2, text_encoder_3 = None, None, None, None, None


    if requires_conversion:
        transformer, vae, text_encoder, text_encoder_2, text_encoder_3 = get_pipeline_components()
    else:
        print("SD3 model already converted")
        return

    if not TRANSFORMER_PATH.exists():
        print("Transformer model conversion started")
        convert_transformer(transformer)
        del transformer
        gc.collect()
        print("Transformer model conversion finished")
    
    else:
        print("Found converted transformer model")

    if not 
