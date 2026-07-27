from typing import Callable
import gradio as gr
from uuid import uuid4
from threading import Thread
from transformers import TextIteratorStreamer


coding_examples = [
    ["Write a quick sort algorithm in Python"],
    ["Explain the difference between a stack and a queue"],
    ["Write a function to check if a string is a palindrome"],
    ["Implement a binary search algorithm in Python"],
    ["Write a Python function to find the factorial of a number"],
    ["Explain what a hash table is and how it works"],
    ["Write a simple linked list implementation in Python"],
    ["Debug this code: def fib(n): return fib(n-1) + fib(n-2)"],
    ["Write a REST API endpoint using Flask"],
    ["Create a Python class for a bank account with deposit and withdraw methods"],
]


def get_uuid():
    return str(uuid4())


def handle_user_message(message, history):
    return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]


def make_demo(model, tokenizer):
    def run_fn(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        if not history:
            return history

        # Build messages from history, skipping the empty assistant placeholder
        messages = [{"role": m["role"], "content": m["content"]} for m in history if m["content"]]

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

        # Streamer for streaming output
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": 2048,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            history[-1]["content"] = generated_text
            yield history

    def stop_fn():
        return None

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown("""<h1><center>Qwen2.5-Coder with OpenVINO</center></h1>""")
        gr.Markdown("""<center><font size=3>Qwen2.5-Coder-7B: Specialized code generation model with OpenVINO optimization</center></center>""")
        chatbot = gr.Chatbot(height=600, type="messages")
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Ask me anything about coding...",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.7,
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                    with gr.Column():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.8,
                            minimum=0.01,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info="Sample from the smallest possible set of tokens whose cumulative probability exceeds top_p.",
                        )
                    with gr.Column():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=20,
                            minimum=0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens.",
                        )
                    with gr.Column():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.0,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.05,
                            interactive=True,
                            info="Penalize repetition - 1.0 to disable.",
                        )
        gr.Examples(coding_examples, inputs=msg, label="Coding Examples")

        submit_event = msg.submit(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=stop_fn,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    return demo
