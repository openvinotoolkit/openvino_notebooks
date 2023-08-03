import gradio as gr
from transformers import pipeline
import time

def transcribe(audio, state=""):
    print(audio)
    time.sleep(2)
    text = pipe(audio)["text"]
    return text, state

def prompt_refinement(text):
    #prompt = "Create an effective prompt for a stable diffusion AI model using the following phrase: " + text
    result = "A beautiful " + text
    return result

def stable_diffusion():
    print("image_gen")

def CLIP():
    print("CLIP")

def trigger_pipeline(audio):
    text,state = transcribe(audio, state)
    text = prompt_refinement(text)
    stable_diffusion()
    CLIP()
    return text, state

with gr.Blocks() as demo:
  state = gr.State(value="")
  audio = gr.Audio(source="microphone", type="filepath")
  textbox = gr.Textbox()
  audio.stream(fn=transcribe, inputs=[audio, state], outputs=[textbox, state])
  textbox = gr.Textbox()


demo.launch(debug=True)