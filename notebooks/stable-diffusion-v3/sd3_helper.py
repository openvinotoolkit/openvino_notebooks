from pathlib import Path

supported_model_ids = [
    "tensorart/stable-diffusion-3.5-medium-turbo",
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3-medium-diffusers",
]


def get_pipeline_options(default_value=(supported_model_ids[0], False)):
    import ipywidgets as widgets

    model_selector = widgets.Dropdown(options=supported_model_ids, value=default_value[0])

    load_t5 = widgets.Checkbox(
        value=default_value[1],
        description="Use t5 text encoder",
        disabled=False,
    )

    to_compress = widgets.Checkbox(
        value=True,
        description="Weight compression",
        disabled=False,
    )

    pt_pipeline_options = widgets.VBox([model_selector, load_t5, to_compress])
    return pt_pipeline_options, model_selector, load_t5, to_compress


def init_pipeline_without_t5(model_dir, device):
    import openvino_genai as ov_genai

    model_path = Path(model_dir)

    scheduler = ov_genai.Scheduler.from_config(model_path / "scheduler/scheduler_config.json")
    text_encoder = ov_genai.CLIPTextModelWithProjection(model_path / "text_encoder", device)
    text_encoder_2 = ov_genai.CLIPTextModelWithProjection(model_path / "text_encoder_2", device)
    transformer = ov_genai.SD3Transformer2DModel(model_path / "transformer", device)
    vae = ov_genai.AutoencoderKL(model_path / "vae_decoder", device=device)

    ov_pipe = ov_genai.Text2ImagePipeline.stable_diffusion_3(scheduler, text_encoder, text_encoder_2, transformer, vae)
    return ov_pipe
