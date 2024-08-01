# main.py
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import generate_adu 
from PIL import Image # Importing the generate_audio module

# Define a request model for FastAPI
class TextPrompt(BaseModel):
    prompt: str

app = FastAPI()

@app.post("/generate-audio/")
async def generate_audio_endpoint(prompt: TextPrompt):
    result = generate_adu.generate(prompt.prompt)
    return {"audio": result.getvalue()}  # Return binary content for the audio file

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_streamlit():
    # Load and display logo
    logo = Image.open("Music_logo.png")
    
    # Display logo and title
    st.image(logo, width=80, use_column_width=False)
    st.title("Sangeet Guru")
    st.subheader("Input Your Style, Get Your Music")

    prompt = st.text_input("Enter your music style:")

    if st.button("Generate"):
        audio_io = generate_adu.generate(prompt)
        st.audio(audio_io, format='audio/wav')

    # Example prompts
    st.markdown("### Examples:")
    examples = [
        "80s pop track with bassy drums and synth",
        "Earthy tones, environmentally conscious, ukulele-infused",
        "90s rock song with loud guitars and heavy drums",
        "Heartful EDM with beautiful synths and chords",
        "Classical Indian raga with sitar and tabla"
    ]

    for example in examples:
        if st.button(example):
            audio_io = generate_adu.generate(example)
            st.audio(audio_io, format='audio/wav')

if __name__ == "__main__":
    # Uncomment the following line to run FastAPI
    # run_fastapi()
    # Uncomment the following line to run Streamlit
    run_streamlit()
