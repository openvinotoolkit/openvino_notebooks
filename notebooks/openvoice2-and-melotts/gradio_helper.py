from typing import Callable
import gradio as gr


description = """
    # OpenVoice2 accelerated by OpenVINO:
    
    a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
"""

content = """
<div>
<strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/docs/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
</div>
"""
wrapped_markdown_content = f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"


examples = [
    [
        "Did you ever hear a folk tale about a giant turtle?",
        "en_latest",
        "OpenVoice/resources/demo_speaker0.mp3",
        True,
    ],
    [
        "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
        "zh_default",
        "OpenVoice/resources/demo_speaker1.mp3",
        True,
    ],
]


def make_demo(fn: Callable):
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            gr.HTML(wrapped_markdown_content)

        with gr.Row():
            with gr.Column():
                input_text_gr = gr.Textbox(
                    label="Text Prompt",
                    info="One or two sentences at a time is better. Up to 50 text characters.",
                    value="The bustling city square bustled with street performers, tourists, and local vendors.",
                )
                style_gr = gr.Dropdown(
                    label="Style",
                    info="Select a style of output audio for the synthesised speech. (Chinese only support 'default' now)",
                    choices=[
                        "en_latest",
                        "zh_default",
                    ],
                    max_choices=1,
                    value="en_latest",
                )
                ref_gr = gr.Audio(
                    label="Reference Audio",
                    # info="Click on the button to upload your own target speaker audio",
                    type="filepath",
                    value="OpenVoice/resources/demo_speaker0.mp3",
                )
                tos_gr = gr.Checkbox(
                    label="Agree",
                    value=False,
                    info="I agree to the terms of the MIT license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
                )

                tts_button = gr.Button("Send", elem_id="send-btn", visible=True)

            with gr.Column():
                out_text_gr = gr.Text(label="Info")
                audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
                ref_audio_gr = gr.Audio(label="Reference Audio Used")

                gr.Examples(
                    examples,
                    label="Examples",
                    inputs=[input_text_gr, style_gr, ref_gr, tos_gr],
                    outputs=[out_text_gr, audio_gr, ref_audio_gr],
                    fn=fn,
                    cache_examples=False,
                )
                tts_button.click(
                    fn,
                    [input_text_gr, style_gr, ref_gr, tos_gr],
                    outputs=[out_text_gr, audio_gr, ref_audio_gr],
                )
    return demo
