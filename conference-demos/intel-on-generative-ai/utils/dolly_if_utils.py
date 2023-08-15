from threading import Thread
from time import perf_counter
from typing import List
from transformers import AutoTokenizer, TextIteratorStreamer
import numpy as np

def get_special_token_id(tokenizer: AutoTokenizer, RESPONSE_KEY, END_KEY):
    """
    Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    """
    tokenizer_response_key = next((token for token in tokenizer.additional_special_tokens if    token.startswith(RESPONSE_KEY)), None)
    end_key_token_id = None
    if tokenizer_response_key:
        try:
            token_ids = tokenizer.encode(END_KEY)
            if len(token_ids) > 1:
                raise ValueError(f"Expected only a single token for '{END_KEY}' but found {token_ids}")
            end_key_token_id = token_ids[0]
            return end_key_token_id
            # Ensure generation stops once it generates "### End"
        except ValueError:
            pass

def run_generation_regular(ov_model, tokenizer, user_text:str, gen_prompt_format:str, top_p:float, temperature:float, top_k:int, max_new_tokens:int, end_key_token_id:int):
    """
    Text generation function
    
    Parameters:
      user_text (str): User-provided instruction for a generation.
      top_p (float):  Nucleus sampling. If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for a generation.
      temperature (float): The value used to module the logits distribution.
      top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
      max_new_tokens (int): Maximum length of generated sequence.
      perf_text (str): Content of text field for printing performance results.
    Returns:
      model_output (str) - model-generated text
      perf_text (str) - updated perf text filed content
    """
    
    # Prepare input prompt according to model expected template
    prompt_text = gen_prompt_format.format(instruction=user_text)
    
    # Tokenize the user text.
    model_inputs = tokenizer(prompt_text, return_tensors="pt")

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    generate_ids = ov_model.generate(model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k,
        eos_token_id=end_key_token_id)
    
    output_text = tokenizer.batch_decode(generate_ids,
                                     skip_special_tokens=True)[0]
    return output_text
    
def run_generation(ov_model, tokenizer, user_text:str, gen_prompt_format:str, top_p:float, temperature:float, top_k:int, max_new_tokens:int, perf_text:str, end_key_token_id:int):
    """
    Text generation function
    
    Parameters:
      user_text (str): User-provided instruction for a generation.
      top_p (float):  Nucleus sampling. If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for a generation.
      temperature (float): The value used to module the logits distribution.
      top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
      max_new_tokens (int): Maximum length of generated sequence.
      perf_text (str): Content of text field for printing performance results.
    Returns:
      model_output (str) - model-generated text
      perf_text (str) - updated perf text filed content
    """
    
    # Prepare input prompt according to model expected template
    prompt_text = gen_prompt_format.format(instruction=user_text)
    
    # Tokenize the user text.
    model_inputs = tokenizer(prompt_text, return_tensors="pt")

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k,
        eos_token_id=end_key_token_id
    )
    t = Thread(target=ov_model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    model_output = ""
    #per_token_time = []
    num_tokens = 0
    #start = perf_counter()
    for new_text in streamer:
        #current_time = perf_counter() - start
        model_output += new_text
        #perf_text, num_tokens = estimate_latency(current_time, perf_text, new_text, per_token_time, num_tokens)
        yield model_output #, perf_text
        #start = perf_counter()
    return model_output #, perf_text

def estimate_latency(current_time:float, current_perf_text:str, new_gen_text:str, per_token_time:List[float], num_tokens:int):
    """
    Helper function for performance estimation
    
    Parameters:
      current_time (float): This step time in seconds.
      current_perf_text (str): Current content of performance UI field.
      new_gen_text (str): New generated text.
      per_token_time (List[float]): history of performance from previous steps.
      num_tokens (int): Total number of generated tokens.
      
    Returns:
      update for performance text field
      update for a total number of tokens
    """
    num_current_toks = len(tokenizer.encode(new_gen_text))
    num_tokens += num_current_toks
    per_token_time.append(num_current_toks / current_time)
    if len(per_token_time) > 10 and len(per_token_time) % 4 == 0:
        current_bucket = per_token_time[:-10]
        return f"Average generation speed: {np.mean(current_bucket):.2f} tokens/s. Total generated tokens: {num_tokens}", num_tokens
    return current_perf_text, num_tokens

def reset_textbox(instruction:str, response:str, perf:str):
    """
    Helper function for resetting content of all text fields
    
    Parameters:
      instruction (str): Content of user instruction field.
      response (str): Content of model response field.
      perf (str): Content of performance info filed
    
    Returns:
      empty string for each placeholder
    """
    return "", "", ""