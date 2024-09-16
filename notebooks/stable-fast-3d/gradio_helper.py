import os
import tempfile
import time
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Callable

import gradio as gr
import numpy as np
import rembg
import torch
from gradio_litmodel3d import LitModel3D
from PIL import Image

import sf3d.utils as sf3d_utils
from sf3d.system import SF3D

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gradio")

rembg_session = rembg.new_session()

COND_WIDTH = 512
COND_HEIGHT = 512
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

# Cached. Doesn't change
c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH)

generated_files = []

example_files = [os.path.join("stable-fast-3d/demo_files/examples", f) for f in os.listdir("stable-fast-3d/demo_files/examples")]


def run_model(model, input_image, remesh_option, vertex_count, texture_size):
    start = time.time()
    with torch.no_grad():
        with nullcontext():
            model_batch = create_batch(input_image)
            model_batch = {k: v.to("cpu") for k, v in model_batch.items()}
            trimesh_mesh, _glob_dict = model.generate_mesh(model_batch, texture_size, remesh_option, vertex_count)
            trimesh_mesh = trimesh_mesh[0]

    # Create new tmp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")

    trimesh_mesh.export(tmp_file.name, file_type="glb", include_normals=True)
    generated_files.append(tmp_file.name)

    print("Generation took:", time.time() - start, "s")

    return tmp_file.name


def create_batch(input_image: Image) -> dict[str, Any]:
    img_cond = torch.from_numpy(np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32) / 255.0).float().clip(0, 1)
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond)

    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    # Add batch dim
    batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
    return batched


@lru_cache
def checkerboard(squares: int, size: int, min_value: float = 0.5):
    base = np.zeros((squares, squares)) + min_value
    base[1::2, ::2] = 1
    base[::2, 1::2] = 1

    repeat_mult = size // squares
    return base.repeat(repeat_mult, axis=0).repeat(repeat_mult, axis=1)[:, :, None].repeat(3, axis=-1)


def remove_background(input_image: Image) -> Image:
    return rembg.remove(input_image, session=rembg_session)


def resize_foreground(
    image: Image,
    ratio: float,
) -> Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image, mode="RGBA").resize((COND_WIDTH, COND_HEIGHT))
    return new_image


def square_crop(input_image: Image) -> Image:
    # Perform a center square crop
    min_size = min(input_image.size)
    left = (input_image.size[0] - min_size) // 2
    top = (input_image.size[1] - min_size) // 2
    right = (input_image.size[0] + min_size) // 2
    bottom = (input_image.size[1] + min_size) // 2
    return input_image.crop((left, top, right, bottom)).resize((COND_WIDTH, COND_HEIGHT))


def show_mask_img(input_image: Image) -> Image:
    img_numpy = np.array(input_image)
    alpha = img_numpy[:, :, 3] / 255.0
    chkb = checkerboard(32, 512) * 255
    new_img = img_numpy[..., :3] * alpha[:, :, None] + chkb * (1 - alpha[:, :, None])
    return Image.fromarray(new_img.astype(np.uint8), mode="RGB")


def requires_bg_remove(image, fr):
    if image is None:
        return (
            gr.update(visible=False, value="Run"),
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    alpha_channel = np.array(image.getchannel("A"))
    min_alpha = alpha_channel.min()

    if min_alpha == 0:
        print("Already has alpha")
        sqr_crop = square_crop(image)
        fr_res = resize_foreground(sqr_crop, fr)
        return (
            gr.update(value="Run", visible=True),
            sqr_crop,
            fr_res,
            gr.update(value=show_mask_img(fr_res), visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        gr.update(value="Remove Background", visible=True),
        None,
        None,
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def update_foreground_ratio(img_proc, fr):
    foreground_res = resize_foreground(img_proc, fr)
    return (
        foreground_res,
        gr.update(value=show_mask_img(foreground_res)),
    )


def make_demo(model):
    def run_button(
        run_btn,
        input_image,
        background_state,
        foreground_ratio,
        remesh_option,
        vertex_count,
        texture_size,
    ):
        if run_btn == "Run":
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            glb_file: str = run_model(model, background_state, remesh_option.lower(), vertex_count, texture_size)
            if torch.cuda.is_available():
                print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
            elif torch.backends.mps.is_available():
                print("Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB")

            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=glb_file, visible=True),
                gr.update(visible=True),
            )
        elif run_btn == "Remove Background":
            rem_removed = remove_background(input_image)

            sqr_crop = square_crop(rem_removed)
            fr_res = resize_foreground(sqr_crop, foreground_ratio)

            return (
                gr.update(value="Run", visible=True),
                sqr_crop,
                fr_res,
                gr.update(value=show_mask_img(fr_res), visible=True),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

    with gr.Blocks() as demo:
        img_proc_state = gr.State()
        background_remove_state = gr.State()
        gr.Markdown(
            """
        # SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement
    
        **SF3D** is a state-of-the-art method for 3D mesh reconstruction from a single image.
        This demo allows you to upload an image and generate a 3D mesh model from it.
    
        **Tips**
        1. If the image already has an alpha channel, you can skip the background removal step.
        2. You can adjust the foreground ratio to control the size of the foreground object. This can influence the shape
        3. You can select the remeshing option to control the mesh topology. This can introduce artifacts in the mesh on thin surfaces and should be turned off in such cases.
        4. You can upload your own HDR environment map to light the 3D model.
        """
        )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_img = gr.Image(type="pil", label="Input Image", sources="upload", image_mode="RGBA")
                    preview_removal = gr.Image(
                        label="Preview Background Removal",
                        type="pil",
                        image_mode="RGB",
                        interactive=False,
                        visible=False,
                    )

                foreground_ratio = gr.Slider(
                    label="Foreground Ratio",
                    minimum=0.5,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                )

                foreground_ratio.change(
                    update_foreground_ratio,
                    inputs=[img_proc_state, foreground_ratio],
                    outputs=[background_remove_state, preview_removal],
                )

                remesh_option = gr.Radio(
                    choices=["None", "Triangle", "Quad"],
                    label="Remeshing",
                    value="None",
                    visible=True,
                )

                vertex_count_slider = gr.Slider(
                    label="Target Vertex Count",
                    minimum=1000,
                    maximum=20000,
                    value=10000,
                    step=1000,
                    visible=True,
                )

                texture_size = gr.Slider(
                    label="Texture Size",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=256,
                    visible=True,
                )

                run_btn = gr.Button("Run", variant="primary", visible=False)

            with gr.Column():
                output_3d = LitModel3D(
                    label="3D Model",
                    visible=False,
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    tonemapping="aces",
                    contrast=1.0,
                    scale=1.0,
                )
                with gr.Column(visible=False, scale=1.0) as hdr_row:
                    gr.Markdown(
                        """## HDR Environment Map
    
                    Select an HDR environment map to light the 3D model. You can also upload your own HDR environment maps.
                    """
                    )

                    with gr.Row():
                        hdr_illumination_file = gr.File(label="HDR Env Map", file_types=[".hdr"], file_count="single")
                        example_hdris = [os.path.join("stable-fast-3d/demo_files/hdri", f) for f in os.listdir("stable-fast-3d/demo_files/hdri")]
                        hdr_illumination_example = gr.Examples(
                            examples=example_hdris,
                            inputs=hdr_illumination_file,
                        )

                        hdr_illumination_file.change(
                            lambda x: gr.update(env_map=x.name if x is not None else None),
                            inputs=hdr_illumination_file,
                            outputs=[output_3d],
                        )

        examples = gr.Examples(
            examples=example_files,
            inputs=input_img,
        )

        input_img.change(
            requires_bg_remove,
            inputs=[input_img, foreground_ratio],
            outputs=[
                run_btn,
                img_proc_state,
                background_remove_state,
                preview_removal,
                output_3d,
                hdr_row,
            ],
        )

        run_btn.click(
            run_button,
            inputs=[
                run_btn,
                input_img,
                background_remove_state,
                foreground_ratio,
                remesh_option,
                vertex_count_slider,
                texture_size,
            ],
            outputs=[
                run_btn,
                img_proc_state,
                background_remove_state,
                preview_removal,
                output_3d,
                hdr_row,
            ],
        )
    return demo
