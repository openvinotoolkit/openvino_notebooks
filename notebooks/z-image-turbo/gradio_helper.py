import random
import re
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
import gradio as gr
import torch


RES_CHOICES = {
    "1024": [
        "1024x1024 ( 1:1 )",
        "1152x896 ( 9:7 )",
        "896x1152 ( 7:9 )",
        "1152x864 ( 4:3 )",
        "864x1152 ( 3:4 )",
        "1248x832 ( 3:2 )",
        "832x1248 ( 2:3 )",
        "1280x720 ( 16:9 )",
        "720x1280 ( 9:16 )",
        "1344x576 ( 21:9 )",
        "576x1344 ( 9:21 )",
    ],
    "1280": [
        "1280x1280 ( 1:1 )",
        "1440x1120 ( 9:7 )",
        "1120x1440 ( 7:9 )",
        "1472x1104 ( 4:3 )",
        "1104x1472 ( 3:4 )",
        "1536x1024 ( 3:2 )",
        "1024x1536 ( 2:3 )",
        "1536x864 ( 16:9 )",
        "864x1536 ( 9:16 )",
        "1680x720 ( 21:9 )",
        "720x1680 ( 9:21 )",
    ],
    "1536": [
        "1536x1536 ( 1:1 )",
        "1728x1344 ( 9:7 )",
        "1344x1728 ( 7:9 )",
        "1728x1296 ( 4:3 )",
        "1296x1728 ( 3:4 )",
        "1872x1248 ( 3:2 )",
        "1248x1872 ( 2:3 )",
        "2048x1152 ( 16:9 )",
        "1152x2048 ( 9:16 )",
        "2016x864 ( 21:9 )",
        "864x2016 ( 9:21 )",
    ],
}

RESOLUTION_SET = []
for resolutions in RES_CHOICES.values():
    RESOLUTION_SET.extend(resolutions)

EXAMPLE_PROMPTS = [
    ["一位男士和他的贵宾犬穿着配套的服装参加狗狗秀，室内灯光，背景中有观众。"],
    [
        "极具氛围感的暗调人像，一位优雅的中国美女在黑暗的房间里。一束强光通过遮光板，在她的脸上投射出一个清晰的闪电形状的光影，正好照亮一只眼睛。高对比度，明暗交界清晰，神秘感，莱卡相机色调。"
    ],
    [
        "一张中景手机自拍照片拍摄了一位留着长黑发的年轻东亚女子在灯光明亮的电梯内对着镜子自拍。她穿着一件带有白色花朵图案的黑色露肩短上衣和深色牛仔裤。她的头微微倾斜，嘴唇嘟起做亲吻状，非常可爱俏皮。她右手拿着一部深灰色智能手机，遮住了部分脸，后置摄像头镜头对着镜子"
    ],
    [
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    ],
    [
        '''A vertical digital illustration depicting a serene and majestic Chinese landscape, rendered in a style reminiscent of traditional Shanshui painting but with a modern, clean aesthetic. The scene is dominated by towering, steep cliffs in various shades of blue and teal, which frame a central valley. In the distance, layers of mountains fade into a light blue and white mist, creating a strong sense of atmospheric perspective and depth. A calm, turquoise river flows through the center of the composition, with a small, traditional Chinese boat, possibly a sampan, navigating its waters. The boat has a bright yellow canopy and a red hull, and it leaves a gentle wake behind it. It carries several indistinct figures of people. Sparse vegetation, including green trees and some bare-branched trees, clings to the rocky ledges and peaks. The overall lighting is soft and diffused, casting a tranquil glow over the entire scene. Centered in the image is overlaid text. At the top of the text block is a small, red, circular seal-like logo containing stylized characters. Below it, in a smaller, black, sans-serif font, are the words 'Zao-Xiang * East Beauty & West Fashion * Z-Image'. Directly beneath this, in a larger, elegant black serif font, is the word 'SHOW & SHARE CREATIVITY WITH THE WORLD'. Among them, there are "SHOW & SHARE", "CREATIVITY", and "WITH THE WORLD"'''
    ],
    [
        """一张虚构的英语电影《回忆之味》（The Taste of Memory）的电影海报。场景设置在一个质朴的19世纪风格厨房里。画面中央，一位红棕色头发、留着小胡子的中年男子（演员阿瑟·彭哈利根饰）站在一张木桌后，他身穿白色衬衫、黑色马甲和米色围裙，正看着一位女士，手中拿着一大块生红肉，下方是一个木制切菜板。在他的右边，一位梳着高髻的黑发女子（演员埃莉诺·万斯饰）倚靠在桌子上，温柔地对他微笑。她穿着浅色衬衫和一条上白下蓝的长裙。桌上除了放有切碎的葱和卷心菜丝的切菜板外，还有一个白色陶瓷盘、新鲜香草，左侧一个木箱上放着一串深色葡萄。背景是一面粗糙的灰白色抹灰墙，墙上挂着一幅风景画。最右边的一个台面上放着一盏复古油灯。海报上有大量的文字信息。左上角是白色的无衬线字体"ARTISAN FILMS PRESENTS"，其下方是"ELEANOR VANCE"和"ACADEMY AWARD® WINNER"。右上角写着"ARTHUR PENHALIGON"和"GOLDEN GLOBE® AWARD WINNER"。顶部中央是圣丹斯电影节的桂冠标志，下方写着"SUNDANCE FILM FESTIVAL GRAND JURY PRIZE 2024"。主标题"THE TASTE OF MEMORY"以白色的大号衬线字体醒目地显示在下半部分。标题下方注明了"A FILM BY Tongyi Interaction Lab"。底部区域用白色小字列出了完整的演职员名单，包括"SCREENPLAY BY ANNA REID"、"CULINARY DIRECTION BY JAMES CARTER"以及Artisan Films、Riverstone Pictures和Heritage Media等众多出品公司标志。整体风格是写实主义，采用温暖柔和的灯光方案，营造出一种亲密的氛围。色调以棕色、米色和柔和的绿色等大地色系为主。两位演员的身体都在腰部被截断。"""
    ],
    [
        """一张方形构图的特写照片，主体是一片巨大的、鲜绿色的植物叶片，并叠加了文字，使其具有海报或杂志封面的外观。主要拍摄对象是一片厚实、有蜡质感的叶子，从左下角到右上角呈对角线弯曲穿过画面。其表面反光性很强，捕捉到一个明亮的直射光源，形成了一道突出的高光，亮面下显露出平行的精细叶脉。背景由其他深绿色的叶子组成，这些叶子轻微失焦，营造出浅景深效果，突出了前景的主叶片。整体风格是写实摄影，明亮的叶片与黑暗的阴影背景之间形成高对比度。图像上有多处渲染文字。左上角是白色的衬线字体文字"PIXEL-PEEPERS GUILD Presents"。右上角同样是白色衬线字体的文字"[Instant Noodle] 泡面调料包"。左侧垂直排列着标题"Render Distance: Max"，为白色衬线字体。左下角是五个硕大的白色宋体汉字"显卡在...燃烧"。右下角是较小的白色衬线字体文字"Leica Glow™ Unobtanium X-1"，其正上方是用白色宋体字书写的名字"蔡几"。识别出的核心实体包括品牌像素偷窥者协会、其产品线泡面调料包、相机型号买不到™ X-1以及摄影师名字造相。"""
    ],
]


def generate_image(
    pipe,
    prompt,
    resolution="1024x1024",
    seed=42,
    guidance_scale=5.0,
    num_inference_steps=50,
    shift=3.0,
    max_sequence_length=512,
    progress=gr.Progress(track_tqdm=True),
):
    width, height = get_resolution(resolution)

    generator = torch.Generator("cpu").manual_seed(seed)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    pipe.scheduler = scheduler

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=max_sequence_length,
    ).images[0]

    return image


def get_resolution(resolution):
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


def make_demo(ov_pipe):

    def generate(
        prompt,
        resolution="1024x1024 ( 1:1 )",
        seed=42,
        steps=9,
        shift=3.0,
        random_seed=True,
        gallery_images=None,
        enhance=False,
        progress=gr.Progress(track_tqdm=True),
    ):
        """
        Generate an image using the Z-Image model based on the provided prompt and settings.
        This function is triggered when the user clicks the "Generate" button. It processes
        the input prompt (optionally enhancing it), configures generation parameters, and
        produces an image using the Z-Image diffusion transformer pipeline.
        Args:
            prompt (str): Text prompt describing the desired image content
            resolution (str): Output resolution in format "WIDTHxHEIGHT ( RATIO )" (e.g., "1024x1024 ( 1:1 )")
            seed (int): Seed for reproducible generation
            steps (int): Number of inference steps for the diffusion process
            shift (float): Time shift parameter for the flow matching scheduler
            random_seed (bool): Whether to generate a new random seed, if True will ignore the seed input
            gallery_images (list): List of previously generated images to append to (only needed for the Gradio UI)
            enhance (bool): This was Whether to enhance the prompt (DISABLED! Do not use)
            progress (gr.Progress): Gradio progress tracker for displaying generation progress (only needed for the Gradio UI)
        Returns:
            tuple: (gallery_images, seed_str, seed_int)
                - gallery_images: Updated list of generated images including the new image
                - seed_str: String representation of the seed used for generation
                - seed_int: Integer representation of the seed used for generation
        """

        if random_seed:
            new_seed = random.randint(1, 1000000)
        else:
            new_seed = seed if seed != -1 else random.randint(1, 1000000)
        try:
            resolution_str = resolution.split(" ")[0]
        except:
            resolution_str = "1024x1024"

        image = generate_image(
            pipe=ov_pipe,
            prompt=prompt,
            resolution=resolution_str,
            seed=new_seed,
            guidance_scale=0.0,
            num_inference_steps=int(steps + 1),
            shift=shift,
        )

        if gallery_images is None:
            gallery_images = []
        # gallery_images.append(image)
        gallery_images = [image] + gallery_images  # latest output to be at the top of the list

        return gallery_images, str(new_seed), int(new_seed)

    with gr.Blocks(title="Z-Image Demo") as demo:
        gr.Markdown(f"""# Z-Image-Turbo - OpenVINO""")
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here...")
                # PE components (Temporarily disabled)
                # with gr.Row():
                #     enable_enhance = gr.Checkbox(label="Enhance Prompt (DashScope)", value=False)
                #     enhance_btn = gr.Button("Enhance Only")

                with gr.Row():
                    choices = [int(k) for k in RES_CHOICES.keys()]
                    res_cat = gr.Dropdown(value=1024, choices=choices, label="Resolution Category")

                    initial_res_choices = RES_CHOICES["1024"]
                    resolution = gr.Dropdown(value=initial_res_choices[0], choices=RESOLUTION_SET, label="Width x Height (Ratio)")

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    random_seed = gr.Checkbox(label="Random Seed", value=True)

                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=8, step=1, interactive=False)
                    shift = gr.Slider(label="Time Shift", minimum=1.0, maximum=10.0, value=3.0, step=0.1)

                generate_btn = gr.Button("Generate", variant="primary")

                # Example prompts
                gr.Markdown("### 📝 Example Prompts")
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt_input, label=None)

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                )
                used_seed = gr.Textbox(label="Seed Used", interactive=False)

        def update_res_choices(_res_cat):
            if str(_res_cat) in RES_CHOICES:
                res_choices = RES_CHOICES[str(_res_cat)]
            else:
                res_choices = RES_CHOICES["1024"]
            return gr.update(value=res_choices[0], choices=res_choices)

        res_cat.change(update_res_choices, inputs=res_cat, outputs=resolution)

        # PE enhancement button (Temporarily disabled)
        # enhance_btn.click(
        #     prompt_enhance,
        #     inputs=[prompt_input, enable_enhance],
        #     outputs=[prompt_input, final_prompt_output]
        # )

        generate_btn.click(
            generate,
            inputs=[prompt_input, resolution, seed, steps, shift, random_seed, output_gallery],
            outputs=[output_gallery, used_seed, seed],
        )
        return demo
