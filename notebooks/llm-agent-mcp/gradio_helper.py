import os
from typing import List
from qwen_agent.gui.utils import convert_history_to_chatbot
from qwen_agent.llm.schema import Message
from qwen_agent.gui import WebUI


class OpenVINOUI(WebUI):
    def run(
        self,
        messages: List[Message] = None,
        share: bool = False,
        server_name: str = None,
        server_port: int = None,
        concurrency_limit: int = 10,
        enable_mention: bool = False,
        **kwargs
    ):
        self.run_kwargs = kwargs

        from qwen_agent.gui.gradio_dep import gr, mgr, ms

        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        with gr.Blocks(
            css=os.path.join(os.path.dirname(__file__), "assets/appBot.css"),
            theme=customTheme,
        ) as demo:
            history = gr.State([])
            with ms.Application():
                with gr.Row(elem_classes="container"):
                    with gr.Column(scale=4):
                        chatbot = mgr.Chatbot(
                            value=convert_history_to_chatbot(messages=messages),
                            avatar_images=[
                                self.user_config,
                                self.agent_config_list,
                            ],
                            height=850,
                            avatar_image_width=80,
                            flushing=False,
                            show_copy_button=True,
                            latex_delimiters=[
                                {"left": "\\(", "right": "\\)", "display": True},
                                {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                                {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                                {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                                {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                                {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                                {"left": "\\[", "right": "\\]", "display": True},
                            ],
                        )

                        input = mgr.MultimodalInput(
                            placeholder=self.input_placeholder,
                        )
                        audio_input = gr.Audio(sources=["microphone"], type="filepath")

                    with gr.Column(scale=1):
                        if len(self.agent_list) > 1:
                            agent_selector = gr.Dropdown(
                                [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                                label="Agents",
                                info="Select an Agent",
                                value=0,
                                interactive=True,
                            )

                        agent_info_block = self._create_agent_info_block()

                        agent_plugins_block = self._create_agent_plugins_block()

                        if self.prompt_suggestions:
                            gr.Examples(
                                label="Example",
                                examples=self.prompt_suggestions,
                                inputs=[input],
                            )

                    if len(self.agent_list) > 1:
                        agent_selector.change(
                            fn=self.change_agent,
                            inputs=[agent_selector],
                            outputs=[agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )

                    input_promise = input.submit(
                        fn=self.add_text,
                        inputs=[input, audio_input, chatbot, history],
                        outputs=[input, audio_input, chatbot, history],
                        queue=False,
                    )

                    if len(self.agent_list) > 1 and enable_mention:
                        input_promise = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector],
                            [chatbot, history, agent_selector],
                        )
                    else:
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history],
                            [chatbot, history],
                        )

                    input_promise.then(self.flushed, None, [input])

            demo.load(None)

        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share, server_name=server_name, server_port=server_port)

    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_interactive = self.agent_list[agent_index]

        if agent_interactive.function_map:
            capabilities = [key for key in agent_interactive.function_map.keys()]
            return gr.CheckboxGroup(
                label="Plugins",
                value=capabilities,
                choices=capabilities,
                interactive=False,
            )

        else:
            return gr.CheckboxGroup(
                label="Plugins",
                value=[],
                choices=[],
                interactive=False,
            )
