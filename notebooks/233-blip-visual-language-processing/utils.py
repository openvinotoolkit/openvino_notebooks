import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np

# specify variable length axes
TEXT_DECODER_DYNAMIC_AXES = {
    "input_ids": {1: "seq_len"},
    "attention_mask": {1: "seq_len"},
    "encoder_hidden_states": {1: "enc_seq_len"},
    "encoder_attention_mask": {1: "enc_seq_len"}
}

# specify output names, logits is main output of model
TEXT_DECODER_OUTPUT_NAMES = ["logits"]


def get_text_decoder_input_dict():
   # prepare example inputs for ONNX export
   input_ids = torch.tensor([[30522]])  # begin of sequence token id
   attention_mask = torch.tensor([[1]])  # attention mask for input_ids
   encoder_hidden_states = torch.rand((1, 10, 768))  # encoder last hidden state from text_encoder
   encoder_attention_mask = torch.ones((1, 10), dtype=torch.long)  # attention mask for encoder hidden states

   input_dict = {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "encoder_hidden_states": encoder_hidden_states,
      "encoder_attention_mask": encoder_attention_mask
   }
   return input_dict


def get_past_key_values_outs(text_decoder_outs):
   past_key_values_outs = []
   for idx, _ in enumerate(text_decoder_outs["past_key_values"]):
      past_key_values_outs.extend([f"out_past_key_value.{idx}.key", f"out_past_key_value.{idx}.value"])
   return past_key_values_outs


def export_text_decoder_to_onnx(model, model_path):
   input_dict = get_text_decoder_input_dict()

   # past key values outputs are output for caching model hidden state
   text_decoder_outs = model(**input_dict)
   past_key_values_outs = get_past_key_values_outs(text_decoder_outs)

   with torch.no_grad():
      torch.onnx.export(
         model, input_dict, model_path, input_names=list(input_dict),
         output_names=TEXT_DECODER_OUTPUT_NAMES + past_key_values_outs, dynamic_axes=TEXT_DECODER_DYNAMIC_AXES
      )


def export_text_decoder_with_past_to_onnx(model, model_path):
   input_dict = get_text_decoder_input_dict()
   text_decoder_outs = model(**input_dict)

   # extend input dictionary with hidden states from previous step
   input_dict_with_past = {**input_dict, "past_key_values": text_decoder_outs["past_key_values"]}

   # provide names for past_key_value inputs in ONNX model
   past_key_values_outs = get_past_key_values_outs(text_decoder_outs)
   past_inputs = [k.replace("out_", "in_") for k in past_key_values_outs]

   # extend input names list and dynamic axes with new inputs
   input_names_with_past = list(input_dict) + past_inputs
   dynamic_axes_with_past = {**TEXT_DECODER_DYNAMIC_AXES}
   for k in past_inputs:
      dynamic_axes_with_past[k] = {2: "prev_seq_len"}

   with torch.no_grad():
      torch.onnx.export(
         model, input_dict_with_past, model_path, input_names=input_names_with_past,
         output_names=TEXT_DECODER_OUTPUT_NAMES + past_key_values_outs, dynamic_axes=dynamic_axes_with_past
      )


def visualize_results(orig_img:PIL.Image.Image, answer:str, question:str = None):
    """
    Helper function for results visualization

    Parameters:
       orig_img (PIL.Image.Image): original image
       answer (str): model answer in text format.
       question (str, *optional*, None): input question, if not provided answer will be used as caption
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    ax.imshow(np.array(orig_img))
    qa_text = "question: {}\nanswer: {}"
    cap_text = "caption: {}"
    ax.set_title(qa_text.format(question, answer) if question is not None else cap_text.format(answer),
                 y=-0.01, pad=-30 if question is not None else -15)
    return fig
