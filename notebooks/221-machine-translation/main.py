import streamlit as st
import time
import numpy as np
import itertools
from openvino.runtime import Core
from tokenizers import SentencePieceBPETokenizer

# Download the machine translation model
# Make sure to download the model using the terminal before running the script
# omz_downloader --name machine-translation-nar-en-de-0002

# Initialize OpenVINO core
core = Core()

# Load the translation model
model = core.read_model('intel/machine-translation-nar-en-de-0002/FP32/machine-translation-nar-en-de-0002.xml')
compiled_model = core.compile_model(model)
input_name = "tokens"
output_name = "pred"
max_tokens = model.input(input_name).shape[1]

# Load tokenizers
src_tokenizer = SentencePieceBPETokenizer.from_file(
    'intel/machine-translation-nar-en-de-0002/tokenizer_src/vocab.json',
    'intel/machine-translation-nar-en-de-0002/tokenizer_src/merges.txt'
)
tgt_tokenizer = SentencePieceBPETokenizer.from_file(
    'intel/machine-translation-nar-en-de-0002/tokenizer_tgt/vocab.json',
    'intel/machine-translation-nar-en-de-0002/tokenizer_tgt/merges.txt'
)

def translate(sentence: str) -> str:
    sentence = sentence.strip()
    assert len(sentence) > 0
    tokens = src_tokenizer.encode(sentence).ids
    tokens = [src_tokenizer.token_to_id('<s>')] + tokens + [src_tokenizer.token_to_id('</s>')]
    pad_length = max_tokens - len(tokens)

    if pad_length > 0:
        tokens = tokens + [src_tokenizer.token_to_id('<pad>')] * pad_length
    assert len(tokens) == max_tokens, "input sentence is too long"
    encoded_sentence = np.array(tokens).reshape(1, -1)

    enc_translated = compiled_model({input_name: encoded_sentence})
    output_key = compiled_model.output(output_name)
    enc_translated = enc_translated[output_key][0]

    sentence = tgt_tokenizer.decode(enc_translated)

    for s in ['</s>', '<s>', '<pad>']:
        sentence = sentence.replace(s, '')

    sentence = sentence.lower().split()
    sentence = " ".join(key for key, _ in itertools.groupby(sentence))
    return sentence

def run_translator():
    st.title("English to German Translator")
    input_sentence = st.text_input("Enter an English sentence:")
    if st.button("Translate"):
        start_time = time.perf_counter()
        translated = translate(input_sentence)
        end_time = time.perf_counter()
        st.write(f'Translated: {translated}')
        st.write(f'Time: {end_time - start_time:.2f}s')

# Run the Streamlit app
if __name__ == "__main__":
    run_translator()
