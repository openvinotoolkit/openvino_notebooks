import time
import gradio as gr
from openvino.runtime import Core, Tensor
from pathlib import Path
import numpy as np
from collections import namedtuple
from functools import partial
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel import OVStableDiffusionXLImg2ImgPipeline, OVStableDiffusionXLPipeline
from transformers import CLIPProcessor
from transformers import AutoTokenizer
from transformers import AutoConfig
from PIL import Image
from utils.whisper_OV_utils import *
from utils.whisper_preprocess_utils import *
from utils.clip_utils import *

#Define a variable that can store the current state of the code
first_run = True
redpj_tokenizer = None
redpj_model = None
base = None
refiner = None
whisper_model = None

#-----Code for setting up models - only run the first time the app is run-----
def ready_whisper_model():
    #Download whisper base model
    REPO_DIR = Path("whisper")
    if not REPO_DIR.exists():
        raise Exception("whisper dir does not exist and is not installed")
        
    import whisper
    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    pass    
    Parameter = namedtuple('Parameter', ['device'])
    def parameters():
      return iter([Parameter(torch.device('cpu'))]) 
    #Replace encoder and decoder with OpenVINO IR files   
    model.encoder = OpenVINOAudioEncoder(ie, 'whisper_models/whisper_encoder.xml', device=device)
    model.decoder = OpenVINOTextDecoder(ie, 'whisper_models/whisper_decoder.xml', device=device)
    model.decode = partial(decode, model)
    model.parameters = parameters
    model.logits = partial(logits, model)
    return model

def ready_clip_model():
    model_checkpoint = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_checkpoint)
    core = Core()

    text_model = core.read_model("clip/clip-vit-base-patch16_text.xml")
    image_model = core.read_model("clip/clip-vit-base-patch16_image.xml")
    text_model = core.compile_model(model=text_model, device_name=device)
    image_model = core.compile_model(model=image_model, device_name=device)
    return processor, text_model, image_model
    
def ready_sd_refiner_model():    
    model_dir = Path("openvino-sd-xl-base-1.0")
    refiner_model_dir = Path("openvino-sd-xl-refiner-1.0")

    base = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device)
    refiner = OVStableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_dir, device=device)
    
    return base, refiner
    
def ready_redpj_model():
    #prompt = "Create an effective prompt for a stable diffusion AI model using the following phrase: " + text
    model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    redpj_tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    redPJ_REPO_DIR = Path("redpajama_chat_models")
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    redpj_model = OVModelForCausalLM.from_pretrained(redPJ_REPO_DIR, device=device, ov_config=ov_config, config=AutoConfig.from_pretrained(redPJ_REPO_DIR))
    return redpj_tokenizer, redpj_model 
    
#--------Pre- and post-processing of outputs from the models-----
def clip_update_utils(processor, text_model, image_model, image, query, n_iters, min_crop_size):
    im_tensor = np.array(image)
    x_dim, y_dim = image.size
    text_inputs = dict(
        processor(text=[query], images=[im_tensor], return_tensors="np")
    )
    image_inputs = text_inputs.pop("pixel_values")

    text_embeds = text_model(text_inputs)[text_model.output()]
    image_embeds = image_model(image_inputs)[image_model.output()]

    initial_similarity = cosine_similarity(text_embeds, image_embeds)
    saliency_map = np.zeros((y_dim, x_dim))

    for _ in tqdm.tqdm(range(n_iters)):
        x, y, crop_size = get_random_crop_params(y_dim, x_dim, min_crop_size)
        im_crop = get_cropped_image(im_tensor, x, y, crop_size)

        image_inputs = processor(images=[im_crop], return_tensors="np").pixel_values
        image_embeds = image_model(image_inputs)[image_model.output()]

        similarity = cosine_similarity(text_embeds, image_embeds) - initial_similarity
        update_saliency_map(saliency_map, similarity, x, y, crop_size)

    fig = plot_saliency_map(im_tensor, saliency_map, query, return_fig=True)
    return fig


def prompt_refinement(transcribed_txt):
    if first_run is True:
      global redpj_tokenizer
      global redpj_model
      redpj_tokenizer, redpj_model = ready_redpj_model()
    prompt = f"<human>: Write a prompt for a art generating AI model with the phrase '{transcribed_txt}' \
    Your answer should be a single, artistic sentence that adds text to the specified phrase.\n<bot>:"
    inputs = redpj_tokenizer(prompt, return_tensors='pt').to(redpj_model.device)
    input_length = inputs.input_ids.shape[1]
    t1 = time.perf_counter()
    outputs = redpj_model.generate(
        **inputs, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
    )
    t2 = time.perf_counter()
    print(f"It took {t2 - t1}s")
    token = outputs.sequences[0, input_length:]
    output_str = redpj_tokenizer.decode(token)
    output_str = output_str.split(".")[0]
    return output_str
    
def sd_image_gen(refined_text):
    #Run SDXL Refiner Model
    if first_run is True:
      global base
      global refiner
      base, refiner = ready_sd_refiner_model()
    latents = base(refined_text, num_inference_steps=15, height=512, width=512, 
               generator=np.random.RandomState(314), output_type="latent").images[0]

    image = refiner(prompt=refined_text, image=latents[None, :], num_inference_steps=15, 
                    generator=np.random.RandomState(314)).images[0]
    
    image.save("sd_result.png")
    
    return image

def clip_image_gen(sd_img, query=None):
    global first_run
    if query == None:
      #Pick a default query
      query = "Where is the brightest part of the image?"
    image = Image.open("sd_result.png")
    n_iters = 300
    min_crop_size = 50
    if first_run is True:
      global processor
      global text_model
      global image_model
      processor, text_model, image_model = ready_clip_model()
    saliency_image = clip_update_utils(processor, text_model, image_model, image, query, n_iters, min_crop_size)
    first_run = False
    return query, saliency_image

def transcribe(audio):
    if first_run is True:
      global whisper_model
      whisper_model = ready_whisper_model()
    audio = resample_wav(audio)
    transcription = whisper_model.transcribe(audio, beam_size=5, best_of=5, task="translate")
    srt_lines = prepare_srt(transcription)
    return "".join(srt_lines)
    

def get_device_value(device_dropdown):
    global device
    device=str(device_dropdown)
    
with gr.Blocks() as demo:
  
  gr.Markdown(
    """
    # Think, Say, See it! Multi-model Gen AI w/ Intel HW & OpenVINO
    This notebook demonstrates how to chain multiple Generative AI models together in OpenVINO with runtime in seconds on Intel CPUs and GPUs. The pipeline is composed of four models: Whisper for speech transcription, RedPajama (chat version) for refinement of the generated text, Stable Diffusion XL for using the text as a prompt for image generation, and CLIP to explore interpretability of the generated image. To explore more of the details of the models and the OpenVINO implementation, check out the associated notebook [here](https://github.com/openvinotoolkit/openvino_notebooks/tree/conference-demos/conference-demos/intel-on-generative-ai/final_e2e_genai_pipeline.ipynb).
    """)
  ie = Core()
  
  device_names = [f"{device}: {ie.get_property(device, 'FULL_DEVICE_NAME')}" 
                for device in ie.available_devices]

  gr.Markdown(
    """
    ## Select your device
    """)
  device_dropdown = gr.Dropdown(ie.available_devices, label=f"Select your device. Available devices: {device_names}")  
  gr.Markdown("Note: When running the app for the first time, all models will be compiled. Please expect a runtime of > 2 min. Subsequent fasters will not require compilation \
  will be much quicker.")
  device_dropdown.input(get_device_value, inputs=[device_dropdown])
  gr.Markdown(
  """
  ## Step 1: Speech transcription with Whisper
  Record your voice to get started!
  """)
  audio = gr.Audio(source="microphone", type="filepath")
  with gr.Row():
    btn1 = gr.Button("Submit")
  whisper_textbox = gr.Textbox()
  btn1.click(transcribe, inputs = [audio], outputs=[whisper_textbox])
  
  gr.Markdown(
  """
  ## Step 2: Text gen with RedPajama-INCITE
  """)
  txtgen_textbox = gr.Textbox()
  whisper_textbox.change(prompt_refinement, inputs=whisper_textbox, outputs=txtgen_textbox)
  
  gr.Markdown(
  """
  ## Step 3: Img gen with Stable Diffusion XL Refiner
  """)
  sd_im = gr.Image()
  txtgen_textbox.change(sd_image_gen, inputs=txtgen_textbox, outputs=sd_im)  
  
  gr.Markdown(
  """
  ## Step 4: Img explainability with CLIP
  The query used for explainability.
  """)
  query_clip = gr.Textbox()
  gr.Markdown(
  """
  The saliency map visualization from CLIP
  """)
  clip_im = gr.Plot()
  sd_im.change(clip_image_gen, inputs = sd_im, outputs = [query_clip, clip_im])
  
  #Accept interactive input for the query from CLIP
  #query_clip.change(clip_image_gen, inputs = [sd_im, query_clip], outputs = [clip_im])

demo.launch(server_name='10.3.233.70', server_port=8880, ssl_certfile="cert.pem", ssl_keyfile="key.pem", ssl_verify=False) 