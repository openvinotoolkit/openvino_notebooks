import gradio as gr
from transformers import TextIteratorStreamer
import librosa
from threading import Thread


def make_demo(model, processor):
    def add_text(chatbot, task_history, text_input, audio_input):
        content = []
        if audio_input is not None:
            content.append({"audio": audio_input})
        if text_input:
            content.append({"text": text_input})
        if not content:
            return chatbot, task_history, "", None

        task_history.append({"role": "user", "content": content})

        user_display = text_input or ""
        if audio_input:
            user_display = f"🎤 [Audio] {user_display}".strip()
        chatbot.append({"role": "user", "content": user_display})
        return chatbot, task_history, "", None

    def reset_state():
        return [], []

    def regenerate(chatbot, task_history):
        if task_history and task_history[-1]["role"] == "assistant":
            task_history.pop()
            chatbot.pop()
        if task_history:
            yield from predict(chatbot, task_history)
        else:
            yield chatbot, task_history

    def predict(chatbot, task_history):
        audios = []
        for message in task_history:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele.get("audio") is not None:
                        audios.append(librosa.load(ele["audio"], sr=processor.feature_extractor.sampling_rate)[0])
        text = processor.apply_chat_template(
            [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}] + task_history,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=text, audios=audios if audios else None, return_tensors="pt", padding=True)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {"max_new_tokens": 512, "streamer": streamer, **inputs}

        chatbot.append({"role": "assistant", "content": ""})
        task_history.append({"role": "assistant", "content": [{"text": ""}]})
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            chatbot[-1]["content"] = generated_text
            task_history[-1]["content"][0]["text"] = generated_text
            yield chatbot, task_history

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>OpenVINO Qwen2-Audio-Instruct Bot</center>""")
        chatbot = gr.Chatbot(label="Qwen2-Audio-7B-Instruct", height=750)
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        with gr.Row():
            text_input = gr.Textbox(placeholder="Type your message here...", label="Text", scale=3)
            submit_btn = gr.Button("🚀 Submit (发送)", variant="primary")
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(fn=add_text, inputs=[chatbot, task_history, text_input, audio_input], outputs=[chatbot, task_history, text_input, audio_input]).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        text_input.submit(fn=add_text, inputs=[chatbot, task_history, text_input, audio_input], outputs=[chatbot, task_history, text_input, audio_input]).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        empty_bin.click(reset_state, outputs=[chatbot, task_history], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot, task_history], show_progress=True)

    return demo
