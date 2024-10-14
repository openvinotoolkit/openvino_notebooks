from pathlib import Path
from typing import Callable
import gradio as gr


from PIL import Image
from typing import Callable
import gradio as gr
import requests
from threading import Thread
from transformers import TextIteratorStreamer

example_image_urls = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "baklava.png",
    ),
]
for url, file_name in example_image_urls:
    Image.open(requests.get(url, stream=True).raw).save(file_name)


def make_demo_llava(model, processor):
    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")
        files = message["files"] if isinstance(message, dict) else message.files
        message_text = message["text"] if isinstance(message, dict) else message.text
        if files:
            # message["files"][-1] is a Dict or just a string
            if isinstance(files[-1], dict):
                image = files[-1]["path"]
            else:
                image = files[-1] if isinstance(files[-1], (list, tuple)) else files[-1].path
        else:
            # if there's no image uploaded for this turn, look for images in the past turns
            # kept inside tuples, take the last one
            for hist in history:
                if type(hist[0]) == tuple:
                    image = hist[0][0]
        try:
            if image is None:
                # Handle the case where image is None
                raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")

        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                # pass
                flag = True
                conversation.extend([{"role": "user", "content": []}])
                continue
            if flag == True:
                conversation[0]["content"] = [{"type": "text", "text": f"{user}"}]
                conversation.append({"role": "assistant", "text": assistant})
                flag = False
                continue
            conversation.extend([{"role": "user", "content": [{"type": "text", "text": user}]}, {"role": "assistant", "text": assistant}])

        conversation.append({"role": "user", "content": [{"type": "text", "text": f"{message_text}"}, {"type": "image"}]})
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        print(f"prompt is -\n{prompt}")
        image = Image.open(image)
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        streamer = TextIteratorStreamer(
            processor,
            **{
                "skip_special_tokens": True,
                "skip_prompt": True,
                "clean_up_tokenization_spaces": False,
            },
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer
    
    demo = gr.ChatInterface(
        fn=bot_streaming,
        title="LLaVA OpenVINO Chatbot",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
        ],
        stop_btn=None,
        multimodal=True,
    )
    return demo



def make_demo_videollava(fn: Callable):
    examples_dir = Path("Video-LLaVA/videollava/serve/examples")
    gr.close_all()
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(label="Input Image", type="filepath"),
            gr.Video(label="Input Video"),
            gr.Textbox(label="Question"),
        ],
        outputs=gr.Textbox(lines=10),
        examples=[
            [
                f"{examples_dir}/extreme_ironing.jpg",
                None,
                "What is unusual about this image?",
            ],
            [
                f"{examples_dir}/waterview.jpg",
                None,
                "What are the things I should be cautious about when I visit here?",
            ],
            [
                f"{examples_dir}/desert.jpg",
                None,
                "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_1.mp4",
                "Why is this video funny?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_3.mp4",
                "Can you identify any safety hazards in this video?",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_9.mp4",
                "Describe the video.",
            ],
            [
                None,
                f"{examples_dir}/sample_demo_22.mp4",
                "Describe the activity in the video.",
            ],
            [
                f"{examples_dir}/sample_img_22.png",
                f"{examples_dir}/sample_demo_22.mp4",
                "Are the instruments in the pictures used in the video?",
            ],
            [
                f"{examples_dir}/sample_img_13.png",
                f"{examples_dir}/sample_demo_13.mp4",
                "Does the flag in the image appear in the video?",
            ],
            [
                f"{examples_dir}/sample_img_8.png",
                f"{examples_dir}/sample_demo_8.mp4",
                "Are the image and the video depicting the same place?",
            ],
        ],
        title="Video-LLaVA🚀",
        allow_flagging="never",
    )
    return demo
