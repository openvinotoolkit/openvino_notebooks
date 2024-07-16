from pathlib import Path
import requests
import gradio as gr
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer


def make_demo(model, processor):
    example_image_urls = [
        (
            "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
            "bee.jpg",
        ),
        (
            "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
            "baklava.png",
        ),
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d", "small.png"),
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754", "chart.png"),
    ]

    for url, file_name in example_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)

    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")
        if message["files"]:
            # message["files"][-1] is a Dict or just a string
            if type(message["files"][-1]) == dict:
                image = message["files"][-1]["path"]
            else:
                image = message["files"][-1]
        else:
            # if there's no image uploaded for this turn, look for images in the past turns
            # kept inside tuples, take the last one
            for hist in history:
                if type(hist[0]) == tuple:
                    image = hist[0][0]
        try:
            if image is None:
                # Handle the case where image is None
                raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")

        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                # pass
                flag = True
                conversation.extend([{"role": "user", "content": ""}])
                continue
            if flag == True:
                conversation[0]["content"] = f"<|image_1|>\n{user}"
                conversation.extend([{"role": "assistant", "content": assistant}])
                flag = False
                continue
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

        if len(history) == 0:
            conversation.append({"role": "user", "content": f"<|image_1|>\n{message['text']}"})
        else:
            conversation.append({"role": "user", "content": message["text"]})
        print(f"prompt is -\n{conversation}")
        prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image = Image.open(image)
        inputs = processor(prompt, image, return_tensors="pt")

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
        title="Phi3 Vision 128K Instruct with OpenVINO",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
            {"text": "What is the text saying?", "files": ["./small.png"]},
            {"text": "What does the chart display?", "files": ["./chart.png"]},
        ],
        description="Try the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) from Microsoft wiht OpenVINO. Upload an image and start chatting about it, or simply try one of the examples below. If you won't upload an image, you will receive an error.",
        stop_btn="Stop Generation",
        multimodal=True,
    )

    return demo
