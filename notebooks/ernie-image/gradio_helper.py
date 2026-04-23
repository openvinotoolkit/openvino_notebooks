import random
import re

import gradio as gr
import torch

SIZE_OPTIONS = [
    "1024x1024",
    "848x1264",
    "1264x848",
    "1376x768",
    "768x1376",
    "1200x896",
    "896x1200",
]

EXAMPLE_PROMPTS = [
    [
        "这是一张人像摄影图片。画面中心是一位年轻女性，正面对着镜头。她留着深棕色的长发，"
        "梳成了两条垂在双肩前的麻花辫，发尾用深色发圈扎紧，额前留着轻薄的空气刘海。"
        "她头戴一顶浅驼色的贝雷帽，身穿一件厚实的米白色粗棒针织毛衣。",
        "1024x1024",
        42,
    ],
    [
        "雄伟的雪山，日出时分，超写实摄影",
        "1024x1024",
        42,
    ],
    [
        "A cute cat sitting on a colorful cushion, studio lighting, high quality, detailed fur texture",
        "1024x1024",
        42,
    ],
    [
        "一位男士和他的贵宾犬穿着配套的服装参加狗狗秀，室内灯光，背景中有观众。",
        "848x1264",
        42,
    ],
    [
        "极具氛围感的暗调人像，一位优雅的中国美女在黑暗的房间里。一束强光通过遮光板，"
        "在她的脸上投射出一个清晰的闪电形状的光影，正好照亮一只眼睛。高对比度，明暗交界清晰，"
        "神秘感，莱卡相机色调。",
        "848x1264",
        123,
    ],
]


def get_resolution(resolution):
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


def make_demo(ov_pipe, enable_pe=False):
    """Build Gradio demo for ERNIE-Image-Turbo with OpenVINO.

    Args:
        ov_pipe: OVErnieImagePipeline instance.
        enable_pe: Whether PE (Prompt Enhancer) is available in the pipeline.
    """

    def generate(
        prompt,
        size="1024x1024",
        seed=42,
        random_seed=True,
        use_pe=False,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt.")

        if random_seed:
            seed = random.randint(1, 1_000_000)  # nosec B311 - UI seed, not security

        width, height = get_resolution(size)
        generator = torch.Generator("cpu").manual_seed(int(seed))

        call_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator=generator,
        )
        if enable_pe:
            call_kwargs["use_pe"] = use_pe

        result = ov_pipe(**call_kwargs)
        image = result.images[0]

        revised_text = ""
        if hasattr(result, "revised_prompts") and result.revised_prompts:
            revised_text = result.revised_prompts[0]

        return image, {"revised_text": revised_text, "seed": str(seed)}

    def update_texts(state):
        return state["revised_text"], state["seed"]

    with gr.Blocks(title="ERNIE-Image-Turbo — OpenVINO") as demo:
        gr.Markdown("""
            # 🎨 ERNIE-Image-Turbo — OpenVINO
            Generate high-quality images from text prompts using ERNIE-Image-Turbo accelerated by OpenVINO.
            """)

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here (Chinese or English)...",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    size_dropdown = gr.Dropdown(
                        label="Image Size",
                        choices=SIZE_OPTIONS,
                        value="1024x1024",
                    )
                    seed_number = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                    )
                with gr.Row():
                    random_seed = gr.Checkbox(label="Random Seed", value=True)
                    if enable_pe:
                        use_pe_checkbox = gr.Checkbox(label="Use Prompt Enhancer (PE)", value=True)
                    else:
                        use_pe_checkbox = gr.Checkbox(label="Use Prompt Enhancer (PE)", value=False, interactive=False)

                generate_btn = gr.Button("Generate", variant="primary")

                gr.Markdown("### 📝 Example Prompts")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=[prompt_input, size_dropdown, seed_number],
                    label=None,
                )

            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Output",
                    type="pil",
                    height=512,
                )
                revised_prompt_output = gr.Textbox(
                    label="Revised Prompt (from PE)",
                    lines=4,
                    interactive=False,
                )
                used_seed = gr.Textbox(label="Seed Used", interactive=False)

        result_state = gr.State()

        generate_btn.click(
            generate, inputs=[prompt_input, size_dropdown, seed_number, random_seed, use_pe_checkbox], outputs=[output_image, result_state]
        ).then(
            update_texts,
            inputs=[result_state],
            outputs=[revised_prompt_output, used_seed],  # instant, no progress
        )

    return demo
