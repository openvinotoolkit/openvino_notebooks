from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def download_models():
    print("Downloading SpeechT5Processor...")
    SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    print("Downloading SpeechT5ForTextToSpeech...")
    SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    print("Downloading SpeechT5HifiGan...")
    SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
