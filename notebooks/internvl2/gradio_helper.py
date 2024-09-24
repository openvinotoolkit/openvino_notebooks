from pathlib import Path
import requests
import gradio as gr
from PIL import Image
import torch
from threading import Thread
from transformers import TextIteratorStreamer
import os
import dataclasses
import base64
import copy
import hashlib
import datetime
from io import BytesIO
from PIL import Image
from typing import Any, List, Dict, Union
from dataclasses import field
from internvl2_helper import build_transform, dynamic_preprocess


def pil2base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def resize_img(img: Image.Image, max_len: int, min_len: int) -> Image.Image:
    max_hw, min_hw = max(img.size), min(img.size)
    aspect_ratio = max_hw / min_hw
    # max_len, min_len = 800, 400
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = img.size
    if H > W:
        H, W = longest_edge, shortest_edge
    else:
        H, W = shortest_edge, longest_edge
    return img.resize((W, H))


block_css = """
.gradio-container {margin: 0.1% 1% 0 1% !important; max-width: 98% !important;};
#buttons button {
    min-width: min(120px,100%);
}
.gradient-text {
    font-size: 28px;
    width: auto;
    font-weight: bold;
    background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}
.plain-text {
    font-size: 22px;
    width: auto;
    font-weight: bold;
}
"""

js = """
function createWaveAnimation() {
    const text = document.getElementById('text');
    var i = 0;
    setInterval(function() {
        const colors = [
            'red, orange, yellow, green, blue, indigo, violet, purple',
            'orange, yellow, green, blue, indigo, violet, purple, red',
            'yellow, green, blue, indigo, violet, purple, red, orange',
            'green, blue, indigo, violet, purple, red, orange, yellow',
            'blue, indigo, violet, purple, red, orange, yellow, green',
            'indigo, violet, purple, red, orange, yellow, green, blue',
            'violet, purple, red, orange, yellow, green, blue, indigo',
            'purple, red, orange, yellow, green, blue, indigo, violet',
        ];
        const angle = 45;
        const colorIndex = i % colors.length;
        text.style.background = `linear-gradient(${angle}deg, ${colors[colorIndex]})`;
        text.style.webkitBackgroundClip = 'text';
        text.style.backgroundClip = 'text';
        text.style.color = 'transparent';
        text.style.fontSize = '28px';
        text.style.width = 'auto';
        text.textContent = 'InternVL2';
        text.style.fontWeight = 'bold';
        i += 1;
    }, 200);
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    // console.log(url_params);
    // console.log('hello world...');
    // console.log(window.location.search);
    // console.log('hello world...');
    // alert(window.location.search)
    // alert(url_params);
    return url_params;
}
"""

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    roles: List[str] = field(
        default_factory=lambda: [
            Conversation.SYSTEM,
            Conversation.USER,
            Conversation.ASSISTANT,
        ]
    )
    mandatory_system_message = "我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    system_message: str = "请尽可能详细地回答用户的问题。"
    messages: List[Dict[str, Any]] = field(default_factory=lambda: [])
    max_image_limit: int = 4
    skip_next: bool = False
    streaming_placeholder: str = "▌"

    def get_system_message(self):
        return self.mandatory_system_message + "\n\n" + self.system_message

    def set_system_message(self, system_message: str):
        self.system_message = system_message
        return self

    def get_prompt(self, include_image=False):
        send_messages = [{"role": "system", "content": self.get_system_message()}]
        # send_messages = []
        for message in self.messages:
            if message["role"] == self.USER:
                user_message = {
                    "role": self.USER,
                    "content": message["content"],
                }
                if include_image and "image" in message:
                    user_message["image"] = []
                    for image in message["image"]:
                        user_message["image"].append(image)
                send_messages.append(user_message)
            elif message["role"] == self.ASSISTANT:
                send_messages.append({"role": self.ASSISTANT, "content": message["content"]})
            elif message["role"] == self.SYSTEM:
                send_messages.append(
                    {
                        "role": self.SYSTEM,
                        "content": message["content"],
                    }
                )
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        return send_messages

    def append_message(
        self,
        role,
        content,
        image_list=None,
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "image": [] if image_list is None else image_list,
                # "filenames": save_filenames,
            }
        )

    def get_images(
        self,
        return_copy=False,
        return_base64=False,
        source: Union[str, None] = None,
    ):
        assert source in [self.USER, self.ASSISTANT, None], f"Invalid source: {source}"
        images = []
        for i, msg in enumerate(self.messages):
            if source and msg["role"] != source:
                continue

            for image in msg.get("image", []):
                # org_image = [i.copy() for i in image]
                if return_copy:
                    image = image.copy()

                if return_base64:
                    image = pil2base64(image)

                images.append(image)

        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, msg in enumerate(self.messages):
            if msg["role"] == self.SYSTEM:
                continue

            alt_str = "user upload image" if msg["role"] == self.USER else "output image"
            image = msg.get("image", [])
            if not isinstance(image, list):
                images = [image]
            else:
                images = image

            img_str_list = []
            for i in range(len(images)):
                image = resize_img(
                    images[i],
                    400,
                    800,
                )
                img_b64_str = pil2base64(image)
                W, H = image.size
                img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt_str}" style="width: {W}px; max-width:none; max-height:none"></img>'
                img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt_str}" />'
                img_str_list.append(img_str)

            if msg["role"] == self.USER:
                msg_str = " ".join(img_str_list) + msg["content"]
                ret.append([msg_str, None])
            else:
                msg_str = msg["content"] + " ".join(img_str_list)
                ret[-1][-1] = msg_str
        return ret

    def update_message(self, role, content, image=None, idx=-1):
        assert len(self.messages) > 0, "No message in the conversation."

        idx = (idx + len(self.messages)) % len(self.messages)

        assert self.messages[idx]["role"] == role, f"Role mismatch: {role} vs {self.messages[idx]['role']}"

        self.messages[idx]["content"] = content
        if image is not None:
            if image not in self.messages[idx]["image"]:
                self.messages[idx]["image"] = []
            if not isinstance(image, list):
                image = [image]
            self.messages[idx]["image"].extend(image)

    def return_last_message(self):
        return self.messages[-1]["content"]

    def end_of_current_turn(self):
        assert len(self.messages) > 0, "No message in the conversation."
        assert self.messages[-1]["role"] == self.ASSISTANT, f"It should end with the message from assistant instead of {self.messages[-1]['role']}."

        if self.messages[-1]["content"][-1] != self.streaming_placeholder:
            return

        self.update_message(self.ASSISTANT, self.messages[-1]["content"][:-1], None)

    def copy(self):
        return Conversation(
            mandatory_system_message=self.mandatory_system_message,
            system_message=self.system_message,
            roles=copy.deepcopy(self.roles),
            messages=copy.deepcopy(self.messages),
        )

    def dict(self):
        messages = []
        for message in self.messages:
            images = []
            for image in message.get("image", []):
                filename = self.save_image(image)
                images.append(filename)

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                    "image": images,
                }
            )
            if len(images) == 0:
                messages[-1].pop("image")

        return {
            "mandatory_system_message": self.mandatory_system_message,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": messages,
        }

    def save_image(self, image: Image.Image) -> str:
        t = datetime.datetime.now()
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        filename = os.path.join(
            ".data",
            "serve_images",
            f"{t.year}-{t.month:02d}-{t.day:02d}",
            f"{image_hash}.jpg",
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

        return filename


def make_demo(model, tokenizer):
    example_image_urls = [
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/gallery/astro_on_unicorn.png", "astro_on_unicorn.png"),
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/gallery/prod_9.jpg", "prod_9.jpg"),
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/gallery/prod_12.png", "prod_12.png"),
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/gallery/prod_1.jpeg", "prod_1.jpeg"),
    ]

    logo_image_urls = [
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/assets/assistant.png", "bot.png"),
        ("https://huggingface.co/spaces/OpenGVLab/InternVL/resolve/main/assets/human.png", "human.png"),
    ]

    for url, file_name in example_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)

    for url, file_name in logo_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)

    def init_state(state=None):
        if state is not None:
            del state
        return Conversation()

    def regenerate(state):
        state.update_message(Conversation.ASSISTANT, None, -1)
        prev_human_msg = state.messages[-2]
        if type(prev_human_msg[1]) in (tuple, list):
            prev_human_msg[1] = (*prev_human_msg[1][:2],)
        state.skip_next = False
        textbox = gr.MultimodalTextbox(value=None, interactive=True)
        return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 2

    def clear_history():
        state = init_state()
        textbox = gr.MultimodalTextbox(value=None, interactive=True)
        return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 2

    def add_text(state, message):
        print(f"state: {state}")
        if not state:
            state = init_state()
        images = message.get("files", [])
        text = message.get("text", "").strip()
        textbox = gr.MultimodalTextbox(value=None, interactive=False)
        if len(text) <= 0 and len(images) == 0:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 5
        images = [Image.open(path).convert("RGB") for path in images]

        if len(images) > 0 and len(state.get_images(source=state.USER)) > 0:
            state = init_state(state)
        state.append_message(Conversation.USER, text, images)
        state.skip_next = False
        return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 2

    transform = build_transform(448)
    state = init_state()

    def bot(
        state,
        temperature,
        top_p,
        repetition_penalty,
        max_new_tokens,
        max_input_tiles,
    ):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_config = {
            "num_beams": 1,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": float(temperature) > 0,
            "streamer": streamer,
        }
        prompt = state.get_prompt(include_image=True)
        state.append_message(Conversation.ASSISTANT, state.streaming_placeholder)
        system_message = prompt[0]["content"]
        send_messages = prompt[1:]
        global_image_cnt = 0
        history, pil_images, max_input_tile_list = [], [], []
        for message in send_messages:
            if message["role"] == "user":
                prefix = ""
                if "image" in message:
                    max_input_tile_temp = []
                    for image in message["image"]:
                        pil_images.append(image)
                        prefix += f"Image-{global_image_cnt + 1}: <image>\n\n"
                        global_image_cnt += 1
                        max_input_tile_temp.append(max(1, max_input_tiles // len(message["image"])))
                    if len(max_input_tile_temp) > 0:
                        max_input_tile_list.append(max_input_tile_temp)
                content = prefix + message["content"]
                history.append(
                    [
                        content,
                    ]
                )
            else:
                history[-1].append(message["content"])
        question, history = history[-1][0], history[:-1]

        if global_image_cnt == 1:
            question = question.replace("Image-1: <image>\n\n", "<image>\n")
            history = [[item[0].replace("Image-1: <image>\n\n", "<image>\n"), item[1]] for item in history]

            # Create a new list to store processed sublists
        flattened_list = []
        # Iterate through all but the last sublist in max_input_tile_list and process them
        for sublist in max_input_tile_list[:-1]:
            processed_sublist = [1] * len(sublist)  # Change each element in the sublist to 1
            flattened_list.extend(processed_sublist)  # Flatten the processed sublist and add to the new list
        # If max_input_tile_list is not empty, add the last sublist to the new list
        if max_input_tile_list:
            flattened_list.extend(max_input_tile_list[-1])
        max_input_tile_list = flattened_list
        assert len(max_input_tile_list) == len(pil_images), "The number of max_input_tile_list and pil_images should be the same."

        image_tiles = []
        if len(pil_images) > 0:
            for current_max_input_tiles, pil_image in zip(max_input_tile_list, pil_images):
                tiles = dynamic_preprocess(
                    pil_image,
                    image_size=448,
                    max_num=current_max_input_tiles,
                    use_thumbnail=model.config.use_thumbnail,
                )
                image_tiles += tiles
            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = None

        thread = Thread(
            target=model.chat,
            kwargs=dict(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                history=None,
                return_history=False,
                generation_config=generation_config,
            ),
        )
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            print(generated_text)
            if generated_text.endswith(model.conv_template.sep):
                generated_text = generated_text[: -len(model.conv_template.sep)]
            state.update_message(Conversation.ASSISTANT, generated_text, None)
            yield (
                state,
                state.to_gradio_chatbot(),
                gr.MultimodalTextbox(interactive=True),
            ) + (
                enable_btn,
                enable_btn,
            )

    textbox = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video"],
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    with gr.Blocks(
        title="InternVL-Chat",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=1.5,
                        value=1.1,
                        step=0.02,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=1024,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
                    max_input_tiles = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=12,
                        step=1,
                        interactive=True,
                        label="Max input tiles (control the image size)",
                    )
                gr.Examples(
                    examples=[
                        [
                            {
                                "files": [
                                    "prod_9.jpg",
                                ],
                                "text": "What's at the far end of the image?",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "astro_on_unicorn.png",
                                ],
                                "text": "What does this image mean?",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "prod_12.png",
                                ],
                                "text": "What are the consequences of the easy decisions shown in this image?",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "prod_1.jpeg",
                                ],
                                "text": "Please describe this image.",
                            }
                        ],
                    ],
                    inputs=[textbox],
                )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="InternVL2",
                    height=580,
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=[
                        "human.png",
                        "bot.png",
                    ],
                    bubble_full_width=False,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        regenerate_btn.click(
            regenerate,
            [state],
            [state, chatbot, textbox] + btn_list,
        ).then(
            bot,
            [
                state,
                temperature,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

        textbox.submit(
            add_text,
            [state, textbox],
            [state, chatbot, textbox] + btn_list,
        ).then(
            bot,
            [
                state,
                temperature,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        submit_btn.click(
            add_text,
            [state, textbox],
            [state, chatbot, textbox] + btn_list,
        ).then(
            bot,
            [
                state,
                temperature,
                top_p,
                repetition_penalty,
                max_output_tokens,
                max_input_tiles,
            ],
            [state, chatbot, textbox] + btn_list,
        )

    return demo
