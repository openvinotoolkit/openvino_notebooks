import time
import uuid
from threading import Event

import gradio as gr
import numpy as np
import torch
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video
from tqdm.auto import tqdm


def get_guidance_config(guidance_scale: float = 3.0, audio_guidance_scale: float = 7.0, fast_mode: bool = False) -> dict:
    return {
        "guidance_scale": guidance_scale,
        "audio_guidance_scale": audio_guidance_scale,
        "guidance_rescale": 0.7,
        "audio_guidance_rescale": 0.7,
        "stg_scale": 0.0 if fast_mode else 1.0,
        "audio_stg_scale": 0.0 if fast_mode else 1.0,
        "modality_scale": 1.0 if fast_mode else 3.0,
        "audio_modality_scale": 1.0 if fast_mode else 3.0,
    }


def validate_static_config(generation_args: dict, static_config: dict | None) -> None:
    if static_config is None:
        return
    shape_changed = any(generation_args[key] != static_config[key] for key in ("width", "height", "num_frames", "frame_rate"))
    uses_cfg = generation_args["guidance_scale"] > 1.0 or generation_args["audio_guidance_scale"] > 1.0
    compiled_with_cfg = static_config["guidance_scale"] > 1.0 or static_config["audio_guidance_scale"] > 1.0
    if shape_changed or uses_cfg != compiled_with_cfg:
        raise ValueError("Static shapes fix the clip dimensions, frame rate, and CFG batch size. Reload the pipeline after changing them.")


def format_perf_metrics(metrics) -> str:
    encoder_seconds = sum(metrics.get_text_encoder_infer_duration().values()) / 1000
    transformer_seconds = sum(metrics.raw_metrics.transformer_inference_durations) / 1000
    decoder_seconds = metrics.get_vae_decoder_infer_duration() / 1000
    total_seconds = metrics.get_generate_duration() / 1000
    return (
        f"Text encoding: {encoder_seconds:.2f} s\n"
        f"Transformer inference: {transformer_seconds:.2f} s\n"
        f"Video/audio decoding: {decoder_seconds:.2f} s\n"
        f"Total generation: {total_seconds:.2f} s"
    )


def make_demo(pipe, generation_mode, use_genai=False, generation_args=None, static_config=None):
    is_image_to_video = generation_mode == "image-to-video"
    defaults = {"width": 768, "height": 512, "num_frames": 49, "num_inference_steps": 30, "frame_rate": 24.0, **get_guidance_config()}
    if generation_args is not None:
        defaults.update(generation_args)
    stop_events: dict[str | None, Event] = {}

    def stop(request: gr.Request):
        event = stop_events.get(request.session_hash)
        if event is None:
            return "No generation is running in this session."
        event.set()
        return "Stop requested; waiting for the current denoising step."

    def generate(
        prompt,
        negative_prompt,
        image,
        width,
        height,
        num_frames,
        num_inference_steps,
        seed,
        guidance_scale,
        audio_guidance_scale,
        fast_mode,
        request: gr.Request,
        progress=gr.Progress(track_tqdm=True),
    ):
        if is_image_to_video and image is None:
            raise gr.Error("Provide an input image for image-to-audio-video generation.")

        frame_rate = defaults["frame_rate"]
        generation_args = {
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "num_frames": int(num_frames),
            "frame_rate": frame_rate,
            "num_inference_steps": int(num_inference_steps),
            **get_guidance_config(guidance_scale, audio_guidance_scale, fast_mode),
        }
        try:
            validate_static_config(generation_args, static_config)
        except ValueError as error:
            raise gr.Error(str(error)) from error

        stop_event = Event()
        stop_events[request.session_hash] = stop_event
        start = time.perf_counter()
        try:
            if use_genai:
                import openvino_genai as ov_genai

                with tqdm(total=int(num_inference_steps), desc="Denoising") as progress_bar:

                    def callback(step, num_steps, latent):
                        progress_bar.update(1)
                        return stop_event.is_set()

                    result = pipe.generate(prompt, generator=ov_genai.TorchGenerator(int(seed)), callback=callback, **generation_args)
                if len(result.video.shape) == 0 or 0 in result.video.shape:
                    if stop_event.is_set():
                        return gr.skip(), "Generation cancelled."
                    raise gr.Error("OpenVINO GenAI returned an empty video.")
                video = torch.from_numpy(np.array(result.video.data[0]))
                audio = torch.from_numpy(np.array(result.audio.data[0]))
                audio_sample_rate = result.audio_sample_rate
                status = format_perf_metrics(result.perf_metrics)
            else:
                generation_args.update(
                    {
                        "prompt": prompt,
                        "generator": torch.Generator("cpu").manual_seed(int(seed)),
                        "output_type": "np",
                        "return_dict": False,
                    }
                )
                if is_image_to_video:
                    generation_args.update({"image": image, "image_crf": 0})

                videos, audios = pipe(**generation_args)
                video = videos[0]
                audio = audios[0].float().cpu()
                audio_sample_rate = pipe.vocoder.config.output_sampling_rate
                status = f"Generation: {time.perf_counter() - start:.2f} s (Optimum Intel; GenAI stage metrics are unavailable)."

            output_path = f"ltx2.3_output_{uuid.uuid4().hex[:8]}.mp4"
            encode_video(video, fps=frame_rate, audio=audio, audio_sample_rate=audio_sample_rate, output_path=output_path)
            return output_path, status
        finally:
            stop_events.pop(request.session_hash, None)

    with gr.Blocks() as demo:
        gr.Markdown("# LTX-2.3 synchronized audio-video generation")
        prompt = gr.Textbox(
            label="Prompt",
            value="A cat stretches lazily on a sunny windowsill, purring softly, while birds chirp outside.",
            lines=4,
        )
        negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
        image = gr.Image(label="Input image", type="pil", visible=is_image_to_video)
        if static_config is not None:
            gr.Markdown("Static shapes are enabled: clip dimensions are fixed. Reload the pipeline to change them or to enable/disable CFG.")
        with gr.Row():
            width = gr.Slider(
                minimum=min(64, defaults["width"]),
                maximum=max(1280, defaults["width"]),
                value=defaults["width"],
                step=32,
                label="Width",
                interactive=static_config is None,
            )
            height = gr.Slider(
                minimum=min(64, defaults["height"]),
                maximum=max(704, defaults["height"]),
                value=defaults["height"],
                step=32,
                label="Height",
                interactive=static_config is None,
            )
            num_frames = gr.Slider(
                minimum=min(9, defaults["num_frames"]),
                maximum=max(257, defaults["num_frames"]),
                value=defaults["num_frames"],
                step=8,
                label="Number of frames",
                interactive=static_config is None,
            )
        with gr.Row():
            num_inference_steps = gr.Slider(minimum=1, maximum=50, value=defaults["num_inference_steps"], step=1, label="Inference steps")
            seed = gr.Slider(minimum=0, maximum=2**31 - 1, value=42, step=1, label="Seed")
        with gr.Row():
            guidance_scale = gr.Slider(minimum=1.0, maximum=12.0, value=defaults["guidance_scale"], step=0.1, label="Video CFG")
            audio_guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=defaults["audio_guidance_scale"], step=0.1, label="Audio CFG")
        fast_mode = gr.Checkbox(value=defaults["stg_scale"] == 0.0 and defaults["modality_scale"] == 1.0, label="Fast mode (CFG only)")
        gr.Markdown("Stop is available with OpenVINO GenAI and takes effect between denoising steps.")
        with gr.Row():
            generate_button = gr.Button("Generate")
            stop_button = gr.Button("Stop", interactive=use_genai)
        result = gr.Video(label="Generated audio-video")
        status = gr.Textbox(label="Status / GenAI stage timings", lines=4, interactive=False)

        generate_button.click(
            generate,
            inputs=[prompt, negative_prompt, image, width, height, num_frames, num_inference_steps, seed, guidance_scale, audio_guidance_scale, fast_mode],
            outputs=[result, status],
            concurrency_limit=1,
        )
        stop_button.click(stop, outputs=status, queue=False)

    return demo
