from pathlib import Path
import requests
from PIL import Image
from typing import List
from qwen_agent.llm.schema import CONTENT, ROLE, USER, Message
from qwen_agent.gui.utils import convert_history_to_chatbot
from qwen_agent.gui.gradio import gr, mgr
from qwen_agent.gui import WebUI


openvino_logo = "openvino_logo.png"
openvino_logo_url = "https://cdn-avatars.huggingface.co/v1/production/uploads/1671615670447-6346651be2dcb5422bcd13dd.png"

if not Path(openvino_logo).exists():
    image = Image.open(requests.get(openvino_logo_url, stream=True).raw)
    image.save(openvino_logo)


chatbot_config = {
    "prompt.suggestions": [
        "Based on current weather in London, show me a picture of Big Ben",
        "What is OpenVINO ?",
        "Create an image of pink cat",
        "What is the weather like in New York now ?",
        "How many people live in Canada ?",
    ],
    "agent.avatar": openvino_logo,
    "input.placeholder": "Please input your request here",
}


class OpenVINOUI(WebUI):
    def request_cancel(self):
        self.agent_list[0].llm.ov_model.request.cancel()

    def clear_history(self):
        return []

    def add_text(self, _input, _chatbot, _history):
        _history.append(
            {
                ROLE: USER,
                CONTENT: [{"text": _input}],
            }
        )
        _chatbot.append([_input, None])
        yield gr.update(interactive=False, value=None), _chatbot, _history

    def run(
        self,
        messages: List[Message] = None,
        share: bool = False,
        server_name: str = None,
        server_port: int = None,
        **kwargs,
    ):
        self.run_kwargs = kwargs

        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=".disclaimer {font-variant-caps: all-small-caps;}",
        ) as self.demo:
            gr.Markdown("""<h1><center>OpenVINO Qwen Agent </center></h1>""")
            history = gr.State([])

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = mgr.Chatbot(
                        value=convert_history_to_chatbot(messages=messages),
                        avatar_images=[
                            self.user_config,
                            self.agent_config_list,
                        ],
                        height=900,
                        avatar_image_width=80,
                        flushing=False,
                        show_copy_button=True,
                    )
                    with gr.Column():
                        input = gr.Textbox(
                            label="Chat Message Box",
                            placeholder="Chat Message Box",
                            show_label=False,
                            container=False,
                        )
                    with gr.Column():
                        with gr.Row():
                            submit = gr.Button("Submit", variant="primary")
                            stop = gr.Button("Stop")
                            clear = gr.Button("Clear")
                with gr.Column(scale=1):
                    agent_interactive = self.agent_list[0]
                    capabilities = [key for key in agent_interactive.function_map.keys()]
                    gr.CheckboxGroup(
                        label="Tools",
                        value=capabilities,
                        choices=capabilities,
                        interactive=False,
                    )
            with gr.Row():
                gr.Examples(self.prompt_suggestions, inputs=[input], label="Click on any example and press the 'Submit' button")

            input_promise = submit.click(
                fn=self.add_text,
                inputs=[input, chatbot, history],
                outputs=[input, chatbot, history],
                queue=False,
            )
            input_promise = input_promise.then(
                self.agent_run,
                [chatbot, history],
                [chatbot, history],
            )
            input_promise.then(self.flushed, None, [input])
            stop.click(
                fn=self.request_cancel,
                inputs=None,
                outputs=None,
                cancels=[input_promise],
                queue=False,
            )
            clear.click(lambda: None, None, chatbot, queue=False).then(self.clear_history, None, history)

            self.demo.load(None)

        self.demo.launch(share=share, server_name=server_name, server_port=server_port)


def make_demo(bot):
    return OpenVINOUI(
        bot,
        chatbot_config=chatbot_config,
    )
