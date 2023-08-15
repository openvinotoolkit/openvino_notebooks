from transformers import LlamaTokenizer, TextIteratorStreamer
from threading import Thread

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful prompt engineer who can optimize prompt text for a Stable Diffusion model. Your answers should be a single sentence, and not include any harmful and unethical content. Please ensure your responses are socially unbiased and positive in nature.
"""

def build_inputs(history: list[tuple[str, str]],
                 query: str,
                 system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in history:
        texts.append(
            f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{query.strip()} [/INST]')
    return ''.join(texts)

def generate_iterate(tokenizer, ov_model, prompt: str, max_generated_tokens, top_k, top_p,
                     temperature):
    # Tokenize the user text.
    model_inputs = tokenizer(prompt, return_tensors="pt")

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(model_inputs,
                           streamer=streamer,
                           max_new_tokens=max_generated_tokens,
                           do_sample=True,
                           top_p=top_p,
                           temperature=float(temperature),
                           top_k=top_k,
                           eos_token_id=tokenizer.eos_token_id)
    t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output
    return model_output