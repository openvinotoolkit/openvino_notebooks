import gradio as gr


def make_demo(interface):

    def generate_tts(text, temperature, repetition_penalty, reference_audio, reference_text):

        if reference_audio and reference_text:
            speaker = interface.create_speaker(reference_audio, reference_text)
        else:
            speaker = None

        output = interface.generate(text=text, speaker=speaker, temperature=temperature, repetition_penalty=repetition_penalty)
        output.save("output.wav")
        return "output.wav"

    with gr.Blocks() as demo:
        gr.Markdown("# OuteTTS-0.1-350M Text-to-Speech Demo")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter text here...")
                temperature = gr.Slider(0.1, 1.0, value=0.1, label="Temperature")
                repetition_penalty = gr.Slider(0.5, 2.0, value=1.1, label="Repetition Penalty")

                gr.Markdown(
                    """
    **Note**: For voice cloning, both a reference audio file and its corresponding transcription must be provided.
    If either the audio file or transcription is missing, the model will generate audio with random characteristics."""
                )
                reference_audio = gr.Audio(label="Reference Audio (for voice cloning)", type="filepath")
                reference_text = gr.Textbox(
                    label="Reference Transcription Text (matching the audio)", placeholder="Enter reference text here if using voice cloning"
                )
                submit_button = gr.Button("Generate Speech")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")

        submit_button.click(fn=generate_tts, inputs=[text_input, temperature, repetition_penalty, reference_audio, reference_text], outputs=audio_output)
    return demo
