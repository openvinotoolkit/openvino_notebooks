import gc
import os
from collections import namedtuple
from pathlib import Path
import warnings

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import snapshot_download
import yaml
import openvino as ov
import torch

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline

MODEL_DIR = Path("models")
VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
UNET_PATH = MODEL_DIR / "unet.xml"
DENSEPOSE_PROCESSOR_PATH = MODEL_DIR / "densepose_processor.xml"
SCHP_PROCESSOR_ATR = MODEL_DIR / "schp_processor_atr.xml"
SCHP_PROCESSOR_LIP = MODEL_DIR / "schp_processor_lip.xml"


def convert(model: torch.nn.Module, xml_path: str, example_input):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        with torch.no_grad():
            converted_model = ov.convert_model(model, example_input=example_input)
        ov.save_model(converted_model, xml_path)

        # cleanup memory
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()


class VaeEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return {"latent_parameters": self.vae.encode(x)["latent_dist"].parameters}


class VaeDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents)


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample=None, timestep=None, encoder_hidden_states=None, return_dict=None):
        result = self.unet(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)
        return result


def download_models():
    resume_path = "zhengchong/CatVTON"
    base_model_path = "booksforcharlie/stable-diffusion-inpainting"
    repo_path = snapshot_download(repo_id=resume_path, local_dir=MODEL_DIR)

    pipeline = CatVTONPipeline(base_ckpt=base_model_path, attn_ckpt=repo_path, attn_ckpt_version="mix", use_tf32=True, device="cpu")

    # fix default config to use cpu
    with open(f"{repo_path}/DensePose/densepose_rcnn_R_50_FPN_s1x.yaml", "r") as fp:
        data = yaml.safe_load(fp)

    data["MODEL"].update({"DEVICE": "cpu"})

    with open(f"{repo_path}/DensePose/densepose_rcnn_R_50_FPN_s1x.yaml", "w") as fp:
        yaml.safe_dump(data, fp)

    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cpu",
    )
    return pipeline, mask_processor, automasker


def convert_pipeline_models(pipeline):
    convert(VaeEncoder(pipeline.vae), VAE_ENCODER_PATH, torch.zeros(1, 3, 1024, 768))
    convert(VaeDecoder(pipeline.vae), VAE_DECODER_PATH, torch.zeros(1, 4, 128, 96))
    del pipeline.vae

    inpainting_latent_model_input = torch.rand(2, 9, 256, 96)
    timestep = torch.tensor(0)
    encoder_hidden_states = torch.Tensor(0)
    example_input = (inpainting_latent_model_input, timestep, encoder_hidden_states)

    convert(UNetWrapper(pipeline.unet), UNET_PATH, example_input)
    del pipeline.unet
    gc.collect()


def convert_automasker_models(automasker):
    from detectron2.export import TracingAdapter  # it's detectron2 from CatVTON repo

    def inference(model, inputs):
        # use do_postprocess=False so it returns ROI mask
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]

    tracing_input = [{"image": torch.rand([3, 800, 800], dtype=torch.float32)}]
    warnings.filterwarnings("ignore")
    traceable_model = TracingAdapter(automasker.densepose_processor.predictor.model, tracing_input, inference)

    convert(traceable_model, DENSEPOSE_PROCESSOR_PATH, tracing_input[0]["image"])
    del automasker.densepose_processor.predictor.model

    convert(automasker.schp_processor_atr.model, SCHP_PROCESSOR_ATR, torch.rand([1, 3, 512, 512], dtype=torch.float32))
    convert(automasker.schp_processor_lip.model, SCHP_PROCESSOR_LIP, torch.rand([1, 3, 473, 473], dtype=torch.float32))
    del automasker.schp_processor_atr.model
    del automasker.schp_processor_lip.model
    gc.collect()


class VAEWrapper(torch.nn.Module):
    def __init__(self, vae_encoder, vae_decoder, scaling_factor):
        super().__init__()
        self.vae_enocder = vae_encoder
        self.vae_decoder = vae_decoder
        self.device = "cpu"
        self.dtype = torch.float32
        self.config = namedtuple("VAEConfig", ["scaling_factor"])(scaling_factor)

    def encode(self, pixel_values):
        ov_outputs = self.vae_enocder(pixel_values).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        result = namedtuple("VAE", "latent_dist")(DiagonalGaussianDistribution(parameters=model_outputs.pop("latent_parameters")))

        return result

    def decode(self, latents):
        outs = self.vae_decoder(latents)
        outs = namedtuple("VAE", "sample")(torch.from_numpy(outs[0]))
        return outs


class ConvUnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        outputs = self.unet(
            {
                "sample": sample,
                "timestep": timestep,
            },
        )

        return [torch.from_numpy(outputs[0])]


class ConvDenseposeProcessorWrapper(torch.nn.Module):
    def __init__(self, densepose_processor):
        super().__init__()
        self.densepose_processor = densepose_processor

    def forward(self, sample, **kwargs):
        from detectron2.structures import Instances, Boxes  # it's detectron2 from CatVTON repo

        outputs = self.densepose_processor(sample[0]["image"])
        boxes = outputs[0]
        classes = outputs[1]
        has_mask = len(outputs) >= 5
        scores = outputs[2 if not has_mask else 3]
        print(scores)
        model_input_size = (
            int(outputs[3 if not has_mask else 4][0]),
            int(outputs[3 if not has_mask else 4][1]),
        )
        filtered_detections = scores >= 0
        boxes = Boxes(boxes[filtered_detections])
        scores = scores[filtered_detections]
        classes = classes[filtered_detections]
        out_dict = {"pred_boxes": boxes, "scores": scores, "pred_classes": classes}

        instances = Instances(model_input_size, **out_dict)

        return [{"instances": instances}]


class ConvSchpProcessorWrapper(torch.nn.Module):
    def __init__(self, schp_processor):
        super().__init__()
        self.schp_processor = schp_processor

    def forward(self, image):
        outputs = self.schp_processor(image)

        return torch.from_numpy(outputs[0])


def get_compiled_pipeline(pipeline, core, device, vae_encoder_path, vae_decoder_path, unet_path, vae_scaling_factor):
    compiled_unet = core.compile_model(unet_path, device.value)
    compiled_vae_encoder = core.compile_model(vae_encoder_path, device.value)
    compiled_vae_decoder = core.compile_model(vae_decoder_path, device.value)

    pipeline.vae = VAEWrapper(compiled_vae_encoder, compiled_vae_decoder, vae_scaling_factor)
    pipeline.unet = ConvUnetWrapper(compiled_unet)

    return pipeline


def get_compiled_automasker(automasker, core, device, densepose_processor_path, schp_processor_atr_path, schp_processor_lip_path):
    compiled_densepose_processor = core.compile_model(densepose_processor_path, device.value)
    compiled_schp_processor_atr = core.compile_model(schp_processor_atr_path, device.value)
    compiled_schp_processor_lip = core.compile_model(schp_processor_lip_path, device.value)

    automasker.densepose_processor.predictor.model = ConvDenseposeProcessorWrapper(compiled_densepose_processor)
    automasker.schp_processor_atr.model = ConvSchpProcessorWrapper(compiled_schp_processor_atr)
    automasker.schp_processor_lip.model = ConvSchpProcessorWrapper(compiled_schp_processor_lip)

    return automasker


def get_pipeline_selection_option(is_optimized_pipe_available=False):
    import ipywidgets as widgets

    use_quantized_models = widgets.Checkbox(
        value=is_optimized_pipe_available,
        description="Use quantized models",
        disabled=not is_optimized_pipe_available,
    )
    return use_quantized_models
