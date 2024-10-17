from pathlib import Path

MODEL_DIR = Path("model")

TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder.xml"
TRANSFORMER_OV_PATH = MODEL_DIR / "transformer_ir.xml"
VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"


def get_pipeline_selection_option(optimized_pipe=None):
    import ipywidgets as widgets

    model_available = optimized_pipe is not None
    use_quantized_models = widgets.Checkbox(
        value=model_available,
        description="Use quantized models",
        disabled=not model_available,
    )
    return use_quantized_models
