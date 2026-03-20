import gradio as gr
import numpy as np
import random
import torch


def make_demo(ov_pipe):
    def get_caption_language(prompt):
        ranges = [
            ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
            # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
            # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
        ]
        for char in prompt:
            if any(start <= char <= end for start, end in ranges):
                return "zh"
        return "en"

    def polish_prompt_en(original_prompt):
        magic_prompt = "Ultra HD, 4K, cinematic composition"
        return original_prompt + ", " + magic_prompt

    def polish_prompt_zh(original_prompt):
        magic_prompt = "超清，4K，电影级构图"
        return original_prompt + "，" + magic_prompt

    def rewrite(input_prompt):
        lang = get_caption_language(input_prompt)
        if lang == "zh":
            return polish_prompt_zh(input_prompt)
        elif lang == "en":

            return polish_prompt_en(input_prompt)

    # --- Model Loading ---
    dtype = torch.bfloat16
    device = "cpu"

    # --- UI Constants and Helpers ---
    MAX_SEED = np.iinfo(np.int32).max

    def get_image_size(aspect_ratio):
        """Converts aspect ratio string to width, height tuple."""
        if aspect_ratio == "1:1":
            return 1328, 1328
        elif aspect_ratio == "16:9":
            return 1664, 928
        elif aspect_ratio == "9:16":
            return 928, 1664
        elif aspect_ratio == "4:3":
            return 1472, 1104
        elif aspect_ratio == "3:4":
            return 1104, 1472
        elif aspect_ratio == "3:2":
            return 1584, 1056
        elif aspect_ratio == "2:3":
            return 1056, 1584
        else:
            # Default to 1:1 if something goes wrong
            return 1328, 1328

    def infer(
        prompt,
        seed=42,
        randomize_seed=False,
        aspect_ratio="16:9",
        guidance_scale=4.0,
        num_inference_steps=50,
        prompt_enhance=True,
        progress=gr.Progress(track_tqdm=True),
    ):
        """
        Generates an image using the local Qwen-Image diffusers pipeline.
        """
        # Hardcode the negative prompt as requested
        negative_prompt = "text, watermark, copyright, blurry, low resolution"

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)  # nosec B311 - UI seed for image generation, not security

        # Convert aspect ratio to width and height
        width, height = get_image_size(aspect_ratio)

        # Set up the generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"Calling pipeline with prompt: '{prompt}'")
        if prompt_enhance:
            prompt = rewrite(prompt)
        print(f"Actual Prompt: '{prompt}'")
        print(f"Negative Prompt: '{negative_prompt}'")
        print(f"Seed: {seed}, Size: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")

        # Generate the image
        image = ov_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
            guidance_scale=1.0,  # Use a fixed default for distilled guidance
        ).images[0]

        return image, seed

    # --- Examples and UI Layout ---
    examples = [
        "A capybara wearing a suit holding a sign that reads Hello World",
        "一幅精致细腻的工笔画，画面中心是一株蓬勃生长的红色牡丹，花朵繁茂，既有盛开的硕大花瓣，也有含苞待放的花蕾，层次丰富，色彩艳丽而不失典雅。牡丹枝叶舒展，叶片浓绿饱满，脉络清晰可见，与红花相映成趣。一只蓝紫色蝴蝶仿佛被画中花朵吸引，停驻在画面中央的一朵盛开牡丹上，流连忘返，蝶翼轻展，细节逼真，仿佛随时会随风飞舞。整幅画作笔触工整严谨，色彩浓郁鲜明，展现出中国传统工笔画的精妙与神韵，画面充满生机与灵动之感。",
        "一位身着淡雅水粉色交领襦裙的年轻女子背对镜头而坐，俯身专注地手持毛笔在素白宣纸上书写“通義千問”四个遒劲汉字。古色古香的室内陈设典雅考究，案头错落摆放着青瓷茶盏与鎏金香炉，一缕熏香轻盈升腾；柔和光线洒落肩头，勾勒出她衣裙的柔美质感与专注神情，仿佛凝固了一段宁静温润的旧时光。",
        " 一个可抽取式的纸巾盒子，上面写着'Face, CLEAN & SOFT TISSUE'下面写着'亲肤可湿水'，左上角是品牌名'洁柔'，整体是白色和浅黄色的色调",
        "手绘风格的水循环示意图，整体画面呈现出一幅生动形象的水循环过程图解。画面中央是一片起伏的山脉和山谷，山谷中流淌着一条清澈的河流，河流最终汇入一片广阔的海洋。山体和陆地上绘制有绿色植被。画面下方为地下水层，用蓝色渐变色块表现，与地表水形成层次分明的空间关系。太阳位于画面右上角，促使地表水蒸发，用上升的曲线箭头表示蒸发过程。云朵漂浮在空中，由白色棉絮状绘制而成，部分云层厚重，表示水汽凝结成雨，用向下箭头连接表示降雨过程。雨水以蓝色线条和点状符号表示，从云中落下，补充河流与地下水。整幅图以卡通手绘风格呈现，线条柔和，色彩明亮，标注清晰。背景为浅黄色纸张质感，带有轻微的手绘纹理。",
        '一个会议室，墙上写着"3.14159265-358979-32384626-4338327950"，一个小陀螺在桌上转动',
        "一个咖啡店门口有一个黑板，上面写着通义千问咖啡，2美元一杯，旁边有个霓虹灯，写着阿里巴巴，旁边有个海报，海报上面是一个中国美女，海报下方写着qwen newbee",
        """A young girl wearing school uniform stands in a classroom, writing on a chalkboard. The text "Introducing Qwen-Image, a foundational image generation model that excels in complex text rendering and precise image editing" appears in neat white chalk at the center of the blackboard. Soft natural light filters through windows, casting gentle shadows. The scene is rendered in a realistic photography style with fine details, shallow depth of field, and warm tones. The girl's focused expression and chalk dust in the air add dynamism. Background elements include desks and educational posters, subtly blurred to emphasize the central action. Ultra-detailed 32K resolution, DSLR-quality, soft bokeh effect, documentary-style composition""",
        "Realistic still life photography style: A single, fresh apple resting on a clean, soft-textured surface. The apple is slightly off-center, softly backlit to highlight its natural gloss and subtle color gradients—deep crimson red blending into light golden hues. Fine details such as small blemishes, dew drops, and a few light highlights enhance its lifelike appearance. A shallow depth of field gently blurs the neutral background, drawing full attention to the apple. Hyper-detailed 8K resolution, studio lighting, photorealistic render, emphasizing texture and form.",
    ]

    css = """
    #col-container {
        margin: 0 auto;
        max-width: 1024px;
    }
    """

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(
                '<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">'
            )
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0, variant="primary")

            result = gr.Image(label="Result", show_label=False, type="pil")

            with gr.Accordion("Advanced Settings", open=False):
                # Negative prompt UI element is removed here

                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )

                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():
                    aspect_ratio = gr.Radio(
                        label="Aspect ratio (width:height)",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                        value="16:9",
                    )
                    prompt_enhance = gr.Checkbox(label="Prompt Enhance", value=True)

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=4.0,
                    )

                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=50,
                    )

            gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                prompt,
                # negative_prompt is no longer an input from the UI
                seed,
                randomize_seed,
                aspect_ratio,
                guidance_scale,
                num_inference_steps,
                prompt_enhance,
            ],
            outputs=[result, seed],
        )
    return demo
