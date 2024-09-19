import gradio as gr
import modelscope_studio as mgr
from transformers import TextIteratorStreamer
import librosa
from threading import Thread


def make_demo(model, processor):
    def add_text(chatbot, task_history, input):
        text_content = input.text
        content = []
        if len(input.files) > 0:
            for i in input.files:
                content.append({"audio": i.path})
        if text_content:
            content.append({"text": text_content})
        task_history.append({"role": "user", "content": content})

        chatbot.append(
            [
                {
                    "text": input.text,
                    "files": input.files,
                },
                None,
            ]
        )
        return chatbot, task_history, None

    def add_file(chatbot, task_history, audio_file):
        """Add audio file to the chat history."""
        task_history.append({"role": "user", "content": [{"audio": audio_file.name}]})
        chatbot.append((f"[Audio file: {audio_file.name}]", None))
        return chatbot, task_history

    def reset_user_input():
        """Reset the user input field."""
        return gr.Textbox.update(value="")

    def reset_state():
        """Reset the chat history."""
        return [], []

    def regenerate(chatbot, task_history):
        """Regenerate the last bot response."""
        if task_history and task_history[-1]["role"] == "assistant":
            task_history.pop()
            chatbot.pop()
        if task_history:
            yield predict(chatbot, task_history)
        return chatbot, task_history

    def predict(chatbot, task_history):
        """Generate a response from the model."""

        audios = []
        for message in task_history:
            if isinstance(message["content"], list):
                print(message)
                for ele in message["content"]:
                    if ele.get("audio") is not None:
                        audios.append(librosa.load(ele["audio"], sr=processor.feature_extractor.sampling_rate)[0])
        text = processor.apply_chat_template(
            [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}] + task_history, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {"max_new_tokens": 512, "streamer": streamer, **inputs}
        chatbot.append([None, ""])
        task_history.append({"role": "assistant", "content": [{"text": ""}]})
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            chatbot[-1][-1] = generated_text
            task_history[-1]["content"][0]["text"] = generated_text
            yield chatbot, task_history

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>OpenVINO Qwen2-Audio-Instruct Bot</center>""")
        chatbot = mgr.Chatbot(label="Qwen2-Audio-7B-Instruct", elem_classes="control-height", height=750)
        user_input = mgr.MultimodalInput(
            interactive=True,
            sources=["microphone", "upload"],
            submit_button_props=dict(value="🚀 Submit (发送)"),
            upload_button_props=dict(value="📁 Upload (上传文件)", show_progress=True),
        )
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
        user_input.submit(fn=add_text, inputs=[chatbot, task_history, user_input], outputs=[chatbot, task_history, user_input], concurrency_limit=40).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        empty_bin.click(reset_state, outputs=[chatbot, task_history], show_progress=True, concurrency_limit=40)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot, task_history], show_progress=True, concurrency_limit=40)

    return demo
