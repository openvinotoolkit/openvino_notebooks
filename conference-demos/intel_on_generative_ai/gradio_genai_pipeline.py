import time
import gradio as gr
from openvino.runtime import Core, Tensor
from pathlib import Path
import numpy as np
from collections import namedtuple
from functools import partial
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino import OVStableDiffusionPipeline
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
sd2_model = None
sdxl_model = None
whisper_model = None

#-----Code for setting up models - only run the first time the app is run-----
def ready_whisper_model():
    #Download whisper base model    
    import whisper
    model = whisper.load_model("base")
    model.to("cpu")
    model.eval()
    pass   
    Parameter = namedtuple('Parameter', ['device'])
    def parameters():
      return iter([Parameter(torch.device('cpu'))]) 
    del model.decoder
    del model.encoder
    ie.set_property({'CACHE_DIR': 'whisper_models/'})
    #Replace whisper encoder and decoder with OpenVINO INT8 models
    model = whisper.load_model("base").to("cpu").eval()
    patch_whisper_for_ov_inference(model)
    model.encoder = OpenVINOAudioEncoder(ie, 'whisper_models/whisper_encoder_int8.xml', device="CPU")
    model.decoder = OpenVINOTextDecoder(ie, 'whisper_models/whisper_decoder_int8.xml', device="CPU")
    return model

def ready_redpj_model():
    model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    redpj_tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    redPJ_REPO_DIR = Path("redpajama_chat_models")
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": "redpajama_chat_models/"}
    redpj_model = OVModelForCausalLM.from_pretrained(redPJ_REPO_DIR, device="CPU", ov_config=ov_config, config=AutoConfig.from_pretrained(redPJ_REPO_DIR))
    return redpj_tokenizer, redpj_model 

def ready_sd2_model():   
    model_id = "helenai/stabilityai-stable-diffusion-2-1-base-ov"
    sd_base_model_dir = Path("sd_2-1_base")
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": "sd_2-1_base/"}
    sd_pipe = OVStableDiffusionPipeline.from_pretrained(sd_base_model_dir, ov_config=ov_config, device="GPU")
    sd_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    return sd_pipe

def ready_sdxl_model():
    model_dir = Path("sd-xl-base-1.0")
    ov_config = {"EXECUTION_MODE_HINT": "ACCURACY", 'NUM_STREAMS': '1', "CACHE_DIR": "sd-xl-base-1.0/"}
    sdxl_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device="GPU", ov_config=ov_config)
    return sdxl_pipe

def ready_clip_model():
    model_checkpoint = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_checkpoint)
    core = Core()
    core.set_property({'CACHE_DIR': 'clip/'})
    text_model = core.read_model("clip/clip-vit-base-patch16_text.xml")
    image_model = core.read_model("clip/clip-vit-base-patch16_image.xml")
    text_model = core.compile_model(model=text_model, device_name="CPU")
    image_model = core.compile_model(model=image_model, device_name="CPU")
    return processor, text_model, image_model
    
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

def sd_image_gen(refined_txt, sd_model_selection):
    global sd2_model
    global sdxl_model
    # Generate an image. 
    if first_run is True or sd2_model is None or sdxl_model is None:
      print(sd_model_selection)
      if sd2_model is None and sd_model_selection == "SDv2.1":
        print("Running Stable Diffusion v2.1")
        sd2_model = ready_sd2_model()
      elif sdxl_model is None and sd_model_selection == "SDXL":
        print("Running Stable Difusion XL")
        sdxl_model = ready_sdxl_model()
    
    if sd_model_selection == "SDv2.1":
      model = sd2_model
    elif sd_model_selection == "SDXL":
      model = sdxl_model
    image = model(refined_txt, num_inference_steps=15, height=512, width=512, 
                  generator=np.random.RandomState(314), output_type="pil").images[0]
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
    transcription = whisper_model.transcribe(audio, beam_size=5, best_of=5, task="transcribe")
    srt_lines = prepare_srt(transcription)
    return "".join(srt_lines)
    
with gr.Blocks() as demo:
  
  gr.Markdown(
    """    
    ![Hybrid AI Demo Flow](https://github.com/QData/TextAttack/assets/22090501/eb9a2e7e-504c-4e90-aeca-b3147ab3b0c5)
    
    In this demo, we'll chain multiple Generative AI models together in OpenVINO, with runtime in seconds on Intel CPUs and GPUs! The pipeline is composed of four models: 
    
    1. Whisper for speech transcription (INT8)
    2. RedPajama-INCITE (chat version - 3B parameters) for refinement of the generated text (INT8)
    3. Stable Diffusion (options: v2.1 and XL) for using the text as a prompt for image generation (FP16)
    4. CLIP to explore interpretability of the generated image. 
    
    Whisper, RedPajama-INCITE, and CLIP is run on CPU, while the Stable Diffusion model is run on GPU.
    
    **NOTE:** Before running this demo, please run the associated notebook [here](https://github.com/openvinotoolkit/openvino_notebooks/tree/conference-demos/conference-demos/intel-on-generative-ai/final_e2e_genai_pipeline.ipynb) to download all required model files for the gradio demo. These elements were not included in the gradio app for conciseness.

    The notebook also explores more of the details of the models and the OpenVINO implementation.
    """)
    
  ie = Core()  
  device_names = [f"{device}: {ie.get_property(device, 'FULL_DEVICE_NAME')}" 
                for device in ie.available_devices]

  
  gr.Markdown("# Run the demo")
  gr.Markdown(f"Your available devices: {device_names}")  
  
  gr.Markdown("**Note**: When running the app for the first time, all models will be compiled, so please expect an overall runtime of > 2 min. Subsequent runs will not require compilation and will be much quicker.")
  
  gr.Markdown(
  """
  ## Step 1: Speech transcription with Whisper
  Record your voice to get started!
  """)
  audio = gr.Audio(source="microphone", type="filepath")
  
  whisper_textbox = gr.Textbox(label="Whisper's output")
  examples = ["rocks.wav","mountain.wav","astronaut.wav"]
  gr.Examples(fn=transcribe, examples=examples, inputs = [audio], outputs=[whisper_textbox],
  run_on_click=True)
  
  with gr.Row():
    btn1 = gr.Button("Submit")
    btn1.click(transcribe, inputs = [audio], outputs=[whisper_textbox])
  
  gr.Markdown(
  """
  ## Step 2: Text gen with RedPajama-INCITE (3B)
  """)
  txtgen_textbox = gr.Textbox()
  whisper_textbox.change(prompt_refinement, inputs=whisper_textbox, outputs=txtgen_textbox)
  
  gr.Markdown(
  """
  ## Step 3: Img gen with Stable Diffusion
  """)
  sd_model_selection = gr.Radio(["SDv2.1", "SDXL"], label="Type of Stable Diffusion model", info="Select v2.1 for faster runtime vs. XL for higher quality results", value="SDv2.1")
  sd_im = gr.Image()
  txtgen_textbox.change(sd_image_gen, inputs=[txtgen_textbox, sd_model_selection], outputs=sd_im)  
  
  gr.Markdown(
  """
  ## Step 4: Img explainability with CLIP
  The query used for explainability.
  """)
  query_clip = gr.Textbox(info="The query used for explainability.")
  gr.Markdown(
  """
  The saliency map visualization from CLIP
  """)
  clip_im = gr.Plot()
  sd_im.change(clip_image_gen, inputs = sd_im, outputs = [query_clip, clip_im])

demo.launch(server_name='10.3.233.70', server_port=8880, ssl_certfile="cert.pem", ssl_keyfile="key.pem", ssl_verify=False) 

del first_run, redpj_tokenizer, redpj_model, sd2_model, sdxl_model, whisper_model