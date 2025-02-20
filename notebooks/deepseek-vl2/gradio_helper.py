import io
import sys
import re
import base64
from threading import Thread
import PIL

import gradio as gr
import torch

from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from deepseek_vl2.serve.app_modules.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    wrap_gen_fn,
)
from deepseek_vl2.serve.app_modules.presets import BOX2COLOR
from deepseek_vl2.serve.app_modules.utils import strip_stop_words, parse_ref_bbox, pil_to_base64, display_example

from deepseek_vl2.serve.inference import convert_conversation_to_prompts
from deepseek_vl2.models.conversation import SeparatorStyle

title = """<h1 align="left" style="min-width:200px; margin-top:0;">Chat with OpenVINO DeepSeek-VL2 </h1>"""
description_top = """Special Tokens: `<image>`,     Visual Grounding: `<|ref|>{query}<|/ref|>`,    Grounding Conversation: `<|grounding|>{question}`"""
description = """"""

with open("DeepSeek-VL2/deepseek_vl2/serve/assets/custom.js", "r", encoding="utf-8") as f, open(
    "DeepSeek-VL2/deepseek_vl2/serve/assets/Kelpy-Codos.js", "r", encoding="utf-8"
) as f2:
    customJS = f.read()
    kelpyCodos = f2.read()


def reload_javascript():
    print("Reloading javascript...")
    js = f"<script>{customJS}</script><script>{kelpyCodos}</script>"

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b"</html>", f"{js}</html>".encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse


def parse_ref_bbox(response, image: PIL.Image.Image):
    try:
        image = image.copy()
        image_h, image_w = image.size
        draw = PIL.ImageDraw.Draw(image)

        ref = re.findall(r"<\|ref\|>.*?<\|/ref\|>", response)
        bbox = re.findall(r"<\|det\|>.*?<\|/det\|>", response)
        assert len(ref) == len(bbox)

        if len(ref) == 0:
            return None

        boxes, labels = [], []
        for box, label in zip(bbox, ref):
            box = box.replace("<|det|>", "").replace("<|/det|>", "")
            label = label.replace("<|ref|>", "").replace("<|/ref|>", "")
            box = box[1:-1]
            for onebox in re.findall(r"\[.*?\]", box):
                boxes.append(eval(onebox))
                labels.append(label)

        for indice, (box, label) in enumerate(zip(boxes, labels)):
            box = (
                int(box[0] / 999 * image_h),
                int(box[1] / 999 * image_w),
                int(box[2] / 999 * image_h),
                int(box[3] / 999 * image_w),
            )

            box_color = BOX2COLOR[indice % len(BOX2COLOR.keys())]
            box_width = 3
            draw.rectangle(box, outline=box_color, width=box_width)

            text_x = box[0]
            text_y = box[1] - 20
            text_color = box_color
            font = PIL.ImageFont.truetype("DeepSeek-VL2/deepseek_vl2/serve/assets/simsun.ttc", size=20)
            draw.text((text_x, text_y), label, font=font, fill=text_color)

        # print(f"boxes = {boxes}, labels = {labels}, re-render = {image}")
        return image
    except Exception as e:
        return None


DEPLOY_MODELS = dict()
IMAGE_TOKEN = "<image>"

examples_list = [
    # visual grounding - 1
    [
        ["DeepSeek-VL2/images/visual_grounding_1.jpeg"],
        "<|ref|>The giraffe at the back.<|/ref|>",
    ],
    # visual grounding - 2
    [
        ["DeepSeek-VL2/images/visual_grounding_2.jpg"],
        "找到<|ref|>淡定姐<|/ref|>",
    ],
    # visual grounding - 3
    [
        ["DeepSeek-VL2/images/visual_grounding_3.png"],
        "Find all the <|ref|>Watermelon slices<|/ref|>",
    ],
    # grounding conversation
    [
        ["DeepSeek-VL2/images/grounding_conversation_1.jpeg"],
        "<|grounding|>I want to throw out the trash now, what should I do?",
    ],
    # in-context visual grounding
    [
        ["DeepSeek-VL2/images/incontext_visual_grounding_1.jpeg", "DeepSeek-VL2/images/icl_vg_2.jpeg"],
        "<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image.",
    ],
    # vqa
    [
        ["DeepSeek-VL2/images/vqa_1.jpg"],
        "Describe each stage of this image in detail",
    ],
    # multi-images
    [
        ["DeepSeek-VL2/images/multi_image_1.jpeg", "DeepSeek-VL2/images/multi_image_2.jpeg", "DeepSeek-VL2/images/multi_image_3.jpeg"],
        "能帮我用这几个食材做一道菜吗?",
    ],
]


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


def reset_state():
    return [], [], "Reset Done"


def deepseek_generate(
    conversations, vl_gpt, vl_chat_processor, tokenizer, stop_words, max_length=256, temperature=1.0, top_p=1.0, repetition_penalty=1.1, chunk_size=-1
):
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        pil_images.extend(message["images"])

    prepare_inputs = vl_chat_processor(conversations=conversations, images=pil_images, inference_mode=True, force_batchify=True, system_prompt="")

    return generate(
        vl_gpt,
        tokenizer,
        prepare_inputs,
        max_gen_len=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        stop_words=stop_words,
        chunk_size=chunk_size,
    )


def generate(
    vl_gpt,
    tokenizer,
    prepare_inputs,
    max_gen_len: int = 256,
    temperature: float = 0,
    repetition_penalty=1.1,
    top_p: float = 0.95,
    stop_words=[],
    chunk_size: int = -1,
):
    """Stream the text output from the multimodality model with prompt and image inputs."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    stop_words_ids = [torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    generation_config = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_gen_len,
        do_sample=True,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )

    if temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        generation_config["do_sample"] = False

    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    yield from streamer


def build_demo(ov_model, processor):
    def retry(
        text,
        images,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
    ):
        if len(history) == 0:
            yield (chatbot, history, "Empty context")
            return

        chatbot.pop()
        history.pop()
        text = history.pop()[-1]
        if type(text) is tuple:
            text, image = text

        yield from predict(
            text,
            images,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
        )

    def preview_images(files):
        if files is None:
            return []

        image_paths = []
        for file in files:
            image_paths.append(file.name)
        return image_paths

    def generate_prompt_with_history(text, images, history, vl_chat_processor, tokenizer, max_length=2048):
        """
        Generate a prompt with history for the deepseek application.

        Args:
            text (str): The text prompt.
            images (list[PIL.Image.Image]): The image prompt.
            history (list): List of previous conversation messages.
            tokenizer: The tokenizer used for encoding the prompt.
            max_length (int): The maximum length of the prompt.

        Returns:
            tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
        """
        global IMAGE_TOKEN

        sft_format = "deepseek"
        user_role_ind = 0
        bot_role_ind = 1

        # Initialize conversation
        conversation = vl_chat_processor.new_chat_template()

        if history:
            conversation.messages = history

        if images is not None and len(images) > 0:

            num_image_tags = text.count(IMAGE_TOKEN)
            num_images = len(images)

            if num_images > num_image_tags:
                pad_image_tags = num_images - num_image_tags
                image_tokens = "\n".join([IMAGE_TOKEN] * pad_image_tags)

                # append the <image> in a new line after the text prompt
                text = image_tokens + "\n" + text
            elif num_images < num_image_tags:
                remove_image_tags = num_image_tags - num_images
                text = text.replace(IMAGE_TOKEN, "", remove_image_tags)

            # print(f"prompt = {text}, len(images) = {len(images)}")
            text = (text, images)

        conversation.append_message(conversation.roles[user_role_ind], text)
        conversation.append_message(conversation.roles[bot_role_ind], "")

        # Create a copy of the conversation to avoid history truncation in the UI
        conversation_copy = conversation.copy()

        rounds = len(conversation.messages) // 2

        for _ in range(rounds):
            current_prompt = get_prompt(conversation)
            current_prompt = current_prompt.replace("</s>", "") if sft_format == "deepseek" else current_prompt

            if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
                return conversation_copy

            if len(conversation.messages) % 2 != 0:
                gr.Error("The messages between user and assistant are not paired.")
                return

            try:
                for _ in range(2):  # pop out two messages in a row
                    conversation.messages.pop(0)
            except IndexError:
                gr.Error("Input text processing failed, unable to respond in this round.")
                return None

        gr.Error("Prompt could not be generated within max_length limit.")
        return None

    def to_gradio_chatbot(conv):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(conv.messages[conv.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, images = msg

                    if isinstance(images, list):
                        for j, image in enumerate(images):
                            if isinstance(image, str):
                                with open(image, "rb") as f:
                                    data = f.read()
                                img_b64_str = base64.b64encode(data).decode()
                                image_str = (
                                    f'<img src="data:image/png;base64,{img_b64_str}" ' f'alt="user upload image" style="max-width: 300px; height: auto;" />'
                                )
                            else:
                                image_str = pil_to_base64(image, f"user upload image_{j}", max_size=800, min_size=400)

                            # replace the <image> tag in the message
                            msg = msg.replace(IMAGE_TOKEN, image_str, 1)

                    else:
                        pass

                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_gradio_history(conv):
        """Convert the conversation to gradio history state."""
        return conv.messages[conv.offset :]

    def get_prompt(conv) -> str:
        """Get the prompt for generation."""
        system_prompt = conv.system_template.format(system_message=conv.system_message)
        if conv.sep_style == SeparatorStyle.DeepSeek:
            seps = [conv.sep, conv.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(conv.messages):
                if message:
                    if type(message) is tuple:  # multimodal message
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            return conv.get_prompt()

    def transfer_input(input_text, input_images):
        print("transferring input text and input image")

        return (input_text, input_images, gr.update(value=""), gr.update(value=None), gr.Button(visible=True))

    @wrap_gen_fn
    def predict(
        text,
        images,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
    ):
        """
        Function to predict the response based on the user's input and selected model.

        Parameters:
        user_text (str): The input text from the user.
        user_image (str): The input image from the user.
        chatbot (str): The chatbot's name.
        history (str): The history of the chat.
        top_p (float): The top-p parameter for the model.
        temperature (float): The temperature parameter for the model.
        max_length_tokens (int): The maximum length of tokens for the model.
        max_context_length_tokens (int): The maximum length of context tokens for the model.
        model_select_dropdown (str): The selected model from the dropdown.

        Returns:
        generator: A generator that yields the chatbot outputs, history, and status.
        """
        print("running the prediction function")

        if images is None:
            images = []

        # load images
        pil_images = []
        for img_or_file in images:
            try:
                # load as pil image
                if isinstance(images, PIL.Image.Image):
                    pil_images.append(img_or_file)
                else:
                    image = PIL.Image.open(img_or_file.name).convert("RGB")
                    pil_images.append(image)
            except Exception as e:
                print(f"Error loading image: {e}")

        conversation = generate_prompt_with_history(
            text,
            pil_images,
            history,
            processor,
            processor.tokenizer,
            max_length=max_context_length_tokens,
        )
        all_conv, last_image = convert_conversation_to_prompts(conversation)

        stop_words = conversation.stop_str
        gradio_chatbot_output = to_gradio_chatbot(conversation)

        full_response = ""
        with torch.no_grad():
            for x in deepseek_generate(
                conversations=all_conv,
                vl_gpt=ov_model,
                vl_chat_processor=processor,
                tokenizer=processor.tokenizer,
                stop_words=stop_words,
                max_length=max_length_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            ):
                full_response += x
                response = strip_stop_words(full_response, stop_words)
                conversation.update_last_message(response)
                gradio_chatbot_output[-1][1] = response

                # sys.stdout.write(x)
                # sys.stdout.flush()

                yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

        if last_image is not None:
            # TODO always render the last image's visual grounding image
            vg_image = parse_ref_bbox(response, last_image)
            if vg_image is not None:
                vg_base64 = pil_to_base64(vg_image, f"vg", max_size=800, min_size=400)
                gradio_chatbot_output[-1][1] += vg_base64
                yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

        print("flushed result to gradio")

        yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"

    with open("DeepSeek-VL2/deepseek_vl2/serve/assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_images = gr.State()

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="deepseek_chatbot",
                        show_share_button=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(show_label=False, placeholder="Enter text", container=False)
                    with gr.Column(
                        min_width=70,
                    ):
                        submitBtn = gr.Button("Send")
                    with gr.Column(
                        min_width=70,
                    ):
                        cancelBtn = gr.Button("Stop")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                upload_images = gr.Files(file_types=["image"], show_label=True)
                gallery = gr.Gallery(columns=[3], height="200px", show_label=True)

                upload_images.change(preview_images, inputs=upload_images, outputs=gallery)

                with gr.Tab(label="Parameter Setting") as parameter_row:
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=4096,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )

                    # show images, but not visible
                    show_images = gr.HTML(visible=False)
                    # show_images = gr.Image(type="pil", interactive=False, visible=False)

        def format_examples(examples_list):
            examples = []
            for images, texts in examples_list:
                examples.append([images, display_example(images), texts])

            return examples

        gr.Examples(
            examples=format_examples(examples_list),
            inputs=[upload_images, show_images, text_box],
        )

        gr.Markdown(description)

        input_widgets = [
            input_text,
            input_images,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, upload_images],
            outputs=[input_text, input_images, text_box, upload_images, submitBtn],
            show_progress=True,
        )

        predict_args = dict(
            fn=predict,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        retry_args = dict(
            fn=retry,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        reset_args = dict(fn=reset_textbox, inputs=[], outputs=[text_box, status_display])

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submitBtn.click(**transfer_input_args).then(**predict_args),
        ]

        emptyBtn.click(reset_state, outputs=output_widgets, show_progress=True)
        emptyBtn.click(**reset_args)
        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            output_widgets,
            show_progress=True,
        )

        cancelBtn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    return demo


def make_demo(ov_model, processor):
    demo = build_demo(ov_model, processor)
    reload_javascript()
    return demo
