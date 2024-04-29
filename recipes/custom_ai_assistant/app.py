import argparse
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional

import gradio as gr
import librosa
import numpy as np
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedTokenizer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer

# Global variables initialization
AUDIO_WIDGET_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are Adrishuo - a helpful, respectful, and honest virtual doctor assistant. "
    "Your role is talking to a patient who just came in."
    "Your primary role is to assist in the collection of Symptom information from patients. "
    "Focus solely on gathering symptom details without offering treatment or medical advice."
    "You must only ask follow-up questions based on the patient's initial descriptions to clarify and gather more details about their symtpoms. "
    "You must not attempt to diagnose, treat, or offer health advice. "
    "Ask one and only the symptom related followup questions and keep it short. "
    "You must strictly not suggest or recommend any treatments, including over-the-counter medication. "
    "You must strictly avoid making any assumptions or conclusions about the causes or nature of the patient's symptoms. "
    "You must strictly avoid providing suggestions to manage their symptoms. "
    "Your interactions should be focused solely on understanding and recording the patient's stated symptoms."
    "Do not collect or use any personal information like age, name, contact, gender, etc."
    "Remember, your role is to aid in symptom information collection in a supportive, unbiased, and factually accurate manner. "
    "Your responses should consistently encourage the patient to discuss their symptoms in greater detail while remaining neutral and non-diagnostic."
)

GREET_THE_CUSTOMER = "Please introduce yourself and greet the patient"

NON_HEALTH_QUERY_PROMPT = (
    "You're Adrishuo, a virtual assistant and here to provide support for health-related questions. While I understand you might have other interests, "
    "You're design is specifically tailored to discuss health concerns. If you have any health-related questions or "
    "if there's anything else you're curious about health-wise, You're here to help."
)

SUMMARIZE_THE_CUSTOMER = (
    "You are now required to summarize the patient's exact provided symptoms for the doctor's review. "
    "Strictly do not mention any personal data like age, name, gender, contact, non-health information etc. when summarizing."
    "Warn the patients for immediate medical seeking in case they exhibit symptoms indicative of critical conditions such as heart attacks, strokes, severe allergic reactions, breathing difficulties, high fever with severe symptoms, significant burns, or severe injuries."
    "Summarize the health-related concerns mentioned by the patient in this conversation, focusing only on the information explicitly provided, without adding any assumptions or unrelated symptoms."

)

NEURAL_CHAT_MODEL_TEMPLATE = ("{% if messages[0]['role'] == 'system' %}"
                              "{% set loop_messages = messages[1:] %}"
                              "{% set system_message = messages[0]['content'] %}"
                              "{% else %}"
                              "{% set loop_messages = messages %}"
                              "{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                              "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. "
                              "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}"
                              "{% endif %}"
                              "{{ '### System:\\n' + system_message.strip() + '\\n' }}"
                              "{% for message in loop_messages %}"
                              "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                              "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                              "{% endif %}"
                              "{% set content = message['content'] %}"
                              "{% if message['role'] == 'user' %}"
                              "{{ '### User:\\n' + content.strip() + '\\n' }}"
                              "{% elif message['role'] == 'assistant' %}"
                              "{{ '### Assistant:\\n' + content.strip() + '\\n'}}"
                              "{% endif %}"
                              "{% endfor %}"
                              )

# Initialize Model variables
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[PreTrainedTokenizer] = None
message_template: Optional[str] = None
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None


health_keywords = [
    'pain', 'ache', 'fever', 'chills', 'fatigue', 'weakness', 'dizziness', 'nausea', 'vomiting', 'diarrhea',
    'constipation', 'abdominal pain', 'cramps', 'bloating', 'gas', 'heartburn', 'loss of appetite', 'weight loss',
    'weight gain', 'dehydration', 'urination issues', 'itching', 'rash', 'hives', 'redness', 'swelling', 'bruising',
    'bleeding', 'cough', 'sore throat', 'stuffy nose', 'runny nose', 'sinus pressure', 'headache', 'migraine',
    'vision changes', 'hearing loss', 'tinnitus', 'ear pain', 'dental pain', 'jaw pain', 'shortness of breath',
    'chest pain', 'palpitations', 'fainting', 'seizures', 'numbness', 'tingling', 'paralysis', 'muscle weakness',
    'muscle spasms', 'joint pain', 'stiffness', 'edema', 'insomnia', 'sleepiness', 'anxiety', 'depression',
    'mood swings', 'confusion', 'memory loss', 'hallucinations', 'delusions', 'sweating', 'temperature sensitivity',
    'thirst', 'skin changes', 'hair loss', 'nail changes', 'lymph node enlargement', 'breast lump', 'urinary changes',
    'sexual dysfunction', 'menstrual changes', 'pregnancy', 'injuries', 'burns', 'poisoning', 'allergies', 'infections',
    'chronic diseases', 'acute illnesses', 'screening', 'vaccination', 'health check-up', 'asthma', 'diabetes',
    'hypertension', 'heart disease', 'cancer', 'flu', 'cold', 'allergy', 'eczema', 'psoriasis', 'arthritis', 'anemia',
    'covid', 'coronavirus', 'migraine', 'obesity', 'thyroid', 'influenza', 'stroke', 'heart attack', 'bronchitis',
    'pneumonia', 'tuberculosis', 'malaria', 'dengue', 'chickenpox', 'measles', 'hepatitis', 'hiv', 'aids',
    'cystic fibrosis', 'scoliosis', 'osteoporosis', 'dementia', 'alzheimer', 'parkinson', 'multiple sclerosis',
    'minute', 'hour', 'day', 'minutes', 'hours', 'days', 'morning', 'evening', 'night', 'yesterday', 'today', 'stress', 'anxious', 'irritability',
    'panic attacks', 'sleep problems', 'snoring', 'sleep apnea', 'accident', 'hurt', 'swollen', 'inflammation',
    'infection', 'discharge', 'itch', 'burning', 'discomfort', 'sensitivity', 'soreness', 'dryness', 'odor', 'taste',
    'vision', 'hearing', 'balance', 'coordination', 'appetite', 'thirst', 'temperature', 'fatigue', 'energy', 'mood',
    'concentration', 'memory', 'alertness', 'awareness', 'frequent', 'head', 'arm', 'leg', 'face', 'nose', 'eye', 'ear', 'mouth', 'throat', 'chest', 'back', 'abdomen', 'groin',
    'hand', 'foot', 'finger', 'toe', 'brain', 'heart', 'lung', 'liver', 'stomach', 'kidney', 'bladder', 'spine',
    'muscle', 'bone', 'joint', 'skin', 'hair', 'nail', 'vein', 'artery', 'gland', 'sneeze', 'terrible', 'body', 'frequent', 'sharp', 'body',
]


def is_health_related_query(prompt: str) -> bool:
    if prompt is None:
        return False
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in health_keywords)


def load_asr_model(model_dir: Path) -> None:
    """
    Load automatic speech recognition model and assign it to a global variable

    Params:
        model_dir: dir with the ASR model
    """
    global asr_model, asr_processor

    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device="AUTO")
    asr_processor = AutoProcessor.from_pretrained(model_dir)


def load_chat_model(model_dir: Path) -> None:
    """
    Load chat model and assign it to a global variable

    Params:
        model_dir: dir with the chat model
    """
    global chat_model, chat_tokenizer, message_template

    # load llama model and its tokenizer
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    chat_model = OVModelForCausalLM.from_pretrained(model_dir, device="AUTO", config=AutoConfig.from_pretrained(model_dir), ov_config=ov_config)
    chat_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # neural chat requires different template than specified in the tokenizer
    message_template = NEURAL_CHAT_MODEL_TEMPLATE if ("neural-chat" in model_dir.name) else chat_tokenizer.default_chat_template


def respond(prompt: str, streamer: BaseStreamer | None = None) -> str:
    """
    Respond to the current prompt

    Params:
        prompt: user's prompt
        streamer: if not None will use it to stream tokens
    Returns:
        The chat's response
    """
    start_time = time.time()  # Start time
    # tokenize input text
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    # generate response tokens
    outputs = chat_model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, streamer=streamer)
    token = outputs[0, input_length:]
    # decode tokens into text
    end_time = time.time()  # End time
    print("Chat model response time: {:.2f} seconds".format(end_time - start_time))
    return chat_tokenizer.decode(token).split("</s>")[0]


def get_conversation(history: List[List[str]]) -> str:
    """
    Combines all messages into one string

    Params:
        history: history of the messages (conversation) so far
    Returns:
        All messages combined into one string
    """
    # the conversation must be in that format to use chat template
    conversation = [
        {"role": "system", "content": SYSTEM_CONFIGURATION},
        {"role": "user", "content": GREET_THE_CUSTOMER}
    ]
    # add prompts to the conversation
    for user_prompt, assistant_response in history:
        if user_prompt:
            conversation.append({"role": "user", "content": user_prompt})
        if assistant_response:
            conversation.append({"role": "assistant", "content": assistant_response})

    # use a template specific to the model
    return chat_tokenizer.apply_chat_template(conversation, chat_template=message_template, tokenize=False)


def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    conv = get_conversation([[None, None]])
    return respond(conv)


def chat(history: List[List[str]]) -> List[List[str]]:
    if not history:
        return history  # Handle empty history gracefully

    user_prompt = history[-1][0] if history[-1] else ""

    if not is_health_related_query(user_prompt):
        # Use the NON_HEALTH_QUERY_PROMPT to incorporate the user's query into a more guided response
        non_health_context = NON_HEALTH_QUERY_PROMPT + f" '{user_prompt}'"
        # Format the conversation for the LLM, adjusting it to include the new context
        conversation_formatted = get_conversation(history[:-1] + [[non_health_context, None]])
        # Generate a response using the formatted conversation context
        non_health_response = respond(conversation_formatted).strip().split('\n')[0]
        # Update the latest entry in history with the generated response
        history[-1][1] = non_health_response
        yield history
    else:
        # Handle health-related queries using the model and streaming
        conversation = get_conversation(history)
        chat_streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, timeout=5)
        thread = Thread(target=lambda: respond(conversation, chat_streamer))
        thread.start()

        try:
            for partial_text in chat_streamer:
                if history[-1][1] is None:
                    history[-1][1] = partial_text
                else:
                    history[-1][1] += partial_text
                yield history
        finally:
            thread.join()

    return history


def transcribe(audio: Tuple[int, np.ndarray], conversation: List[List[str]]) -> List[List[str]]:
    """
    Transcribe audio to text

    Params:
        audio: audio to transcribe text from
        conversation: conversation history with the chatbot
    Returns:
        User prompt as a text
    """
    start_time = time.time()  # Start time for ASR process

    sample_rate, audio = audio
    # the whisper model requires 16000Hz, not 44100Hz
    audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=AUDIO_WIDGET_SAMPLE_RATE).astype(np.int16)

    # get input features from the audio
    input_features = asr_processor(audio, sampling_rate=AUDIO_WIDGET_SAMPLE_RATE, return_tensors="pt").input_features
    # get output
    predicted_ids = asr_model.generate(input_features)
    # decode output to text
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    end_time = time.time()  # End time for ASR process
    print(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

    # add the text to the conversation
    conversation.append([transcription, None])
    return conversation


def summarize(conversation: List) -> str:
    """
    Generate a summary of the patient's case based on the conversation so far,
    focusing specifically on the patient's inputs about symptoms.

    Params:
        conversation: history of the messages so far
    Returns:
        A generator yielding a bullet-point summary of the patient's case
    """
    # Extract patient inputs from the conversation
    patient_inputs = "\n- ".join(msg[0] for msg in conversation if msg[0] and 'assistant' not in msg[0])

    # Create a summarization prompt that includes both the directive and the patient's inputs
    summarization_prompt = f"{SUMMARIZE_THE_CUSTOMER}\n\nPatient symptoms:\n- {patient_inputs}"

    # Append this as a new system message to guide the summarization
    conversation.append([summarization_prompt, None])

    # Use the chat function to process this new conversation entry, now including the summary instruction
    summary_generator = chat(conversation)

    # Yield the summary parts as they are generated
    for partial_summary in summary_generator:
        # Split at sentence end for better readability, if needed
        if partial_summary[-1][1]:
            yield partial_summary[-1][1].split("</s>")[0]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Talk to Adrishuo - a custom AI assistant working as a healthcare assistant") as demo:
        gr.Markdown("""
        # Talk to Adrishuo - a custom AI assistant working today as a healthcare assistant

        Instructions for use:
        - record your question/comment using the first audio widget ("Your voice input")
        - wait for the chatbot to response ("Chatbot")
        - click summarize button to make a summary
        """)
        with gr.Row():
            # user's input
            input_audio_ui = gr.Audio(sources=["microphone"], scale=5, label="Your voice input")
            # submit button
            submit_audio_btn = gr.Button("Submit", variant="primary", scale=1, interactive=False)

        # chatbot
        chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")

        # summarize
        summarize_button = gr.Button("Summarize", variant="primary", interactive=False)
        summary_ui = gr.Textbox(label="Summary", interactive=False)

        # events
        # block submit button when no audio input
        input_audio_ui.change(lambda x: gr.Button(interactive=False) if x is None else gr.Button(interactive=True), inputs=input_audio_ui, outputs=submit_audio_btn)

        # block buttons, do the transcription and conversation, clear audio, unblock buttons
        submit_audio_btn.click(lambda: gr.Button(interactive=False), outputs=submit_audio_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button)\
            .then(transcribe, inputs=[input_audio_ui, chatbot_ui], outputs=chatbot_ui)\
            .then(chat, chatbot_ui, chatbot_ui)\
            .then(lambda: None, inputs=[], outputs=[input_audio_ui])\
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        # block button, do the summarization, unblock button
        summarize_button.click(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(summarize, inputs=chatbot_ui, outputs=summary_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

    return demo


def run(asr_model_dir: Path, chat_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the assistant application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        chat_model_dir: dir with the chat model
        public_interface: whether UI should be available publicly
    """
    # load whisper model
    load_asr_model(asr_model_dir)
    # load chat model
    load_chat_model(chat_model_dir)

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.launch(share=public_interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr_model_dir', type=str, default="model/distil-large-v2-FP16", help="Path to the automatic speech recognition model directory")
    parser.add_argument('--chat_model_dir', type=str, default="model/llama2-7B-INT8", help="Path to the chat model directory")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model_dir), Path(args.chat_model_dir), args.public_interface)
