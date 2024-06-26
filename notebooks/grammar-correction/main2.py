import streamlit as st
import transformers
from transformers import pipeline
from pathlib import Path
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM, OVModelForSequenceClassification
from tqdm import tqdm

# Function to split text into sentences
def split_text(text: str) -> list:
    sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)
    sentence_batches = []
    temp_batch = []
    for sentence in sentences:
        temp_batch.append(sentence)
        if len(temp_batch) >= 2 and len(temp_batch) <= 3 or sentence == sentences[-1]:
            sentence_batches.append(temp_batch)
            temp_batch = []
    return sentence_batches

# Function to correct text
def correct_text(text: str, checker, corrector, separator: str = " ") -> str:
    sentence_batches = split_text(text)
    corrected_text = []
    for batch in tqdm(sentence_batches, total=len(sentence_batches), desc="Correcting text..."):
        raw_text = " ".join(batch)
        results = checker(raw_text)
        if results[0]["label"] != "LABEL_1" or (results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9):
            corrected_batch = corrector(raw_text)
            corrected_text.append(corrected_batch[0]["generated_text"])
        else:
            corrected_text.append(raw_text)
    corrected_text = separator.join(corrected_text)
    return corrected_text

# Load models
grammar_checker_model_id = "textattack/roberta-base-CoLA"
grammar_checker_dir = Path("roberta-base-cola")
grammar_checker_tokenizer = AutoTokenizer.from_pretrained(grammar_checker_model_id)

if grammar_checker_dir.exists():
    grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_dir)
else:
    grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_model_id, export=True)
    grammar_checker_model.save_pretrained(grammar_checker_dir)

grammar_checker_pipe = pipeline("text-classification", model=grammar_checker_model, tokenizer=grammar_checker_tokenizer)

grammar_corrector_model_id = "pszemraj/flan-t5-large-grammar-synthesis"
grammar_corrector_dir = Path("flan-t5-large-grammar-synthesis")
grammar_corrector_tokenizer = AutoTokenizer.from_pretrained(grammar_corrector_model_id)

if grammar_corrector_dir.exists():
    grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_dir)
else:
    grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_model_id, export=True)
    grammar_corrector_model.save_pretrained(grammar_corrector_dir)

grammar_corrector_pipe = pipeline("text2text-generation", model=grammar_corrector_model, tokenizer=grammar_corrector_tokenizer)

# Streamlit app
st.set_page_config(page_title="Grammar Guruji", page_icon=":pencil2:")
st.title('GRAMMER GURUJI')
st.markdown(
    """
    <style>
        .title-text {
            font-size: 36px !important;
            color: #3366ff !important;
            text-shadow: 2px 2px 4px #cccccc;
        }
        .subtitle-text {
            font-size: 24px !important;
            color: #4d4d4d !important;
            text-shadow: 1px 1px 2px #cccccc;
        }
        .button-widget {
            background-color: #3366ff !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            box-shadow: 2px 2px 4px #cccccc !important;
        }
        .background {
            background-image: url('https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_960_720.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }
        .stTextInput>div>div>div>input {
            background-color: rgba(255, 255, 255, 0.5) !important;
        }
        .stTextInput>div>div>div>textarea {
            background-color: rgba(255, 255, 255, 0.5) !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

input_text = st.text_area("Input Text")
if st.button('Correct Grammar', key='correct_button', help='Click to correct grammar'):
    with st.spinner('Correcting...'):
        corrected_text = correct_text(input_text, grammar_checker_pipe, grammar_corrector_pipe)
        st.success('Done!')
        st.subheader('Corrected Text')
        st.write(corrected_text)