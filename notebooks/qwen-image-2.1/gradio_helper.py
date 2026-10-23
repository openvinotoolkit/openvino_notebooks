from __future__ import annotations

import gc
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
PIPELINE_CLASSES = {
    "Text-to-Image": ov_genai.Text2ImagePipeline,
    "Image Editing": ov_genai.Image2ImagePipeline,
}
PRECISIONS = ("FP16", "INT8", "INT4")
MAX_CONDITION_IMAGES = 3
EDITING_SAMPLE_PROMPT = "Put a tiny red top hat and a blue bow tie on the cat. " "Preserve the cat's identity, face, pose, and the original composition."
MULTI_IMAGE_SAMPLE_PROMPT = (
    "Place the cat from Picture 1 next to the astronaut from Picture 2 on the lunar surface. "
    "Preserve the cat's appearance, the astronaut's suit, and the moon landscape from Picture 2."
)


def _output_to_image(output: Any) -> Image.Image:
    """Convert an OpenVINO GenAI image output to a PIL image."""
    data = output.data if hasattr(output, "data") else output
    array = np.asarray(data)
    if array.ndim == 4:
        array = array[0]
    return Image.fromarray(array).convert("RGB")


def _image_to_tensor(image: Image.Image) -> ov.Tensor:
    """Convert a PIL image to an NHWC OpenVINO tensor."""
    return ov.Tensor(np.asarray(image.convert("RGB"), dtype=np.uint8)[None])


class PipelineManager:
    """Own the single OpenVINO GenAI pipeline used by the demo."""

    def __init__(self, model_root: str | Path, output_dir: str | Path) -> None:
        self.model_root = Path(model_root)
        self.output_dir = Path(output_dir)
        self.pipeline: Any | None = None
        self.pipeline_key: tuple[str, str, str] | None = None
        self.lock = threading.Lock()

    def _release(self) -> None:
        self.pipeline = None
        self.pipeline_key = None
        gc.collect()

    def release(self) -> str:
        """Release the current pipeline.

        :return: Status message for the demo.
        """
        with self.lock:
            self._release()
        return "Configuration changed. The previous pipeline was released."

    def _load(self, pipeline_type: str, precision: str, device: str) -> tuple[Any, bool]:
        """Load or reuse a pipeline for the selected configuration.

        :param pipeline_type: Text-to-image or image-editing pipeline name.
        :param precision: Exported model precision.
        :param device: OpenVINO inference device.
        :return: Pipeline instance and whether it was newly loaded.
        """
        pipeline_key = (pipeline_type, precision, device)
        if self.pipeline is not None and self.pipeline_key == pipeline_key:
            return self.pipeline, False

        self._release()
        model_dir = self.model_root / precision
        if not model_dir.is_dir():
            raise gr.Error(f"{precision} model was not found in {model_dir}. Export it before running the demo.")

        pipeline_class = PIPELINE_CLASSES[pipeline_type]
        self.pipeline = pipeline_class(str(model_dir), device)
        self.pipeline_key = pipeline_key
        return self.pipeline, True

    def generate(
        self,
        pipeline_type: str,
        precision: str,
        device: str,
        prompt: str,
        seed: int,
        randomize_seed: bool,
        steps: int,
        resolution: int,
        *condition_images: Image.Image | None,
    ) -> tuple[Image.Image, int, str]:
        """Generate and save an image.

        :param pipeline_type: Text-to-image or image-editing pipeline name.
        :param precision: Exported model precision.
        :param device: OpenVINO inference device.
        :param prompt: Generation or editing instruction.
        :param seed: Random generator seed.
        :param randomize_seed: Whether to replace the provided seed.
        :param steps: Number of denoising steps.
        :param resolution: Square output resolution used by text-to-image.
        :param condition_images: Optional Picture 1, Picture 2, ... slots used by image editing.
        :return: Generated image, effective seed, and status message.
        """
        if not prompt.strip():
            raise gr.Error("Enter a prompt or editing instruction.")
        images = [image for image in condition_images if image is not None]
        if pipeline_type == "Image Editing":
            if not images:
                raise gr.Error("Select at least Picture 1 for Image Editing.")
            if any(image is None for image in condition_images[: len(images)]):
                raise gr.Error("Fill the picture slots in order: the prompt refers to them as Picture 1, Picture 2, and so on.")
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)  # nosec B311 - UI seed, not security-sensitive

        with self.lock:
            pipeline, loaded = self._load(pipeline_type, precision, device)
            generation_args = {
                "guidance_scale": 1.0,
                "num_inference_steps": steps,
                "generator": ov_genai.TorchGenerator(seed),
            }
            if pipeline_type == "Text-to-Image":
                generation_args.update(height=resolution, width=resolution)
                output = pipeline.generate(prompt, **generation_args)
            else:
                output = pipeline.generate(prompt, [_image_to_tensor(image) for image in images], **generation_args)

        image = _output_to_image(output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        scenario = "t2i" if pipeline_type == "Text-to-Image" else "editing"
        safe_device = device.replace(":", "-").replace(".", "-")
        output_path = self.output_dir / f"qwen-image-2.1-{scenario}-{precision.lower()}-{safe_device.lower()}-seed{seed}-{timestamp}.png"
        image.save(output_path)

        action = "Loaded" if loaded else "Reused"
        conditioning = f" with {len(images)} condition image(s)" if pipeline_type == "Image Editing" else ""
        status = f"{action} {pipeline_type} {precision} pipeline on {device}{conditioning}. Saved to {output_path.resolve()}."
        return image, seed, status


def make_demo(
    model_root: str | Path,
    default_precision: str = "INT4",
    default_device: str = "CPU",
    output_dir: str | Path = "generated_images",
    editing_sample: str | Path | None = None,
    reference_sample: str | Path | None = None,
) -> gr.Blocks:
    """Create the interactive Qwen-Image 2.1 demo.

    :param model_root: Directory containing exported precision subdirectories.
    :param default_precision: Initially selected model precision.
    :param default_device: Initially selected OpenVINO inference device.
    :param output_dir: Directory used to save generated images.
    :param editing_sample: Optional image used by the editing examples as Picture 1.
    :param reference_sample: Optional image used by the multi-image editing example as Picture 2.
    :return: Configured Gradio Blocks application.
    """
    manager = PipelineManager(model_root=model_root, output_dir=output_dir)
    available_devices = list(dict.fromkeys(["AUTO", *ov.Core().available_devices]))
    if default_device not in available_devices:
        default_device = "CPU" if "CPU" in available_devices else available_devices[0]

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen-Image 2.1 with OpenVINO")
        gr.Markdown(
            "Select a pipeline, exported precision, and inference device. Only one pipeline is kept in memory. "
            "Image editing is multimodal conditioning and does not use a strength parameter. "
            "Up to three condition images can be passed; refer to them as Picture 1, Picture 2, and Picture 3 in the prompt."
        )

        with gr.Row():
            pipeline_type = gr.Dropdown(
                choices=list(PIPELINE_CLASSES),
                value="Text-to-Image",
                label="Pipeline",
            )
            precision = gr.Dropdown(
                choices=list(PRECISIONS),
                value=default_precision,
                label="Precision",
            )
            device = gr.Dropdown(
                choices=available_devices,
                value=default_device,
                label="Device",
            )

        prompt = gr.Textbox(
            label="Prompt or editing instruction",
            value='A cozy coffee shop with a chalkboard sign reading "Qwen Coffee", cinematic lighting',
            lines=3,
        )
        with gr.Row():
            condition_images = [
                gr.Image(
                    type="pil",
                    label=f"Picture {index + 1} ({'required' if index == 0 else 'optional'}, Image Editing only)",
                )
                for index in range(MAX_CONDITION_IMAGES)
            ]
        with gr.Row():
            seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
            randomize_seed = gr.Checkbox(value=False, label="Randomize seed")
        with gr.Row():
            steps = gr.Slider(1, 80, value=40, step=1, label="Steps")
            resolution = gr.Dropdown(
                choices=[512, 768, 1024],
                value=1024,
                label="T2I square resolution",
            )

        run = gr.Button("Run", variant="primary")
        output = gr.Image(label="Generated image")
        status = gr.Textbox(label="Pipeline status", value="No pipeline loaded.", interactive=False)

        if editing_sample is not None:
            editing_sample = Path(editing_sample)
            if not editing_sample.is_file():
                raise FileNotFoundError(f"Image editing sample was not found at {editing_sample}.")
            examples = [["Image Editing", EDITING_SAMPLE_PROMPT, str(editing_sample), None]]
            if reference_sample is not None:
                reference_sample = Path(reference_sample)
                if not reference_sample.is_file():
                    raise FileNotFoundError(f"Multi-image editing sample was not found at {reference_sample}.")
                examples.append(["Image Editing", MULTI_IMAGE_SAMPLE_PROMPT, str(editing_sample), str(reference_sample)])
            gr.Examples(
                examples=examples,
                inputs=[pipeline_type, prompt, *condition_images[:2]],
                label="Image editing examples",
            )

        run.click(
            fn=manager.generate,
            inputs=[
                pipeline_type,
                precision,
                device,
                prompt,
                seed,
                randomize_seed,
                steps,
                resolution,
                *condition_images,
            ],
            outputs=[output, seed, status],
        )
        for selector in (pipeline_type, precision, device):
            selector.change(fn=manager.release, outputs=status, queue=False)

    demo.queue(default_concurrency_limit=1)
    return demo
