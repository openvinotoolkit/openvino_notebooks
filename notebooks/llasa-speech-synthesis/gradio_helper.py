import torch
import gradio as gr


def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def make_demo(ov_model, tokenizer, codec_model):
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    def infer(input_text):
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text
        chat = [{"role": "user", "content": "Convert the text to speech:" + formatted_text}, {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}]

        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
        outputs = ov_model.generate(
            input_ids,
            max_length=2048,
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,  #  Adjusts the diversity of generated content
            temperature=0.8,  #  Controls randomness in output
        )
        # Extract the speech tokens
        generated_ids = outputs[0][input_ids.shape[1] : -1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Convert  token <|s_23456|> to int 23456
        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = torch.tensor(speech_tokens).unsqueeze(0).unsqueeze(0)

        # Decode the speech tokens to speech waveform
        gen_wav = codec_model.decode_code(speech_tokens)
        return (16000, gen_wav[0, 0, :].cpu().numpy())

    with gr.Blocks() as demo:
        gen_text_input = gr.Textbox(label="Text to Generate", lines=10)

        generate_btn = gr.Button("Synthesize", variant="primary")

        audio_output = gr.Audio(label="Synthesized Audio")

        generate_btn.click(
            infer,
            inputs=[
                gen_text_input,
            ],
            outputs=[audio_output],
        )

    return demo
