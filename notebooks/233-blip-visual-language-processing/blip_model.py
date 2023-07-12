import torch
import numpy as np
from openvino.runtime import Model
from typing import List, Tuple, Dict
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def prepare_past_inputs(past_key_values:List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Helper function for rearrange input hidden states inputs to OpenVINO model expected format
    Parameters:
      past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): list of pairs key, value attention hidden states obtained as model outputs from previous step
    Returns:
      inputs (Dict[str, torch.Tensor]): dictionary with inputs for model
    """
    inputs = {}
    for idx, (key, value) in enumerate(past_key_values):
        inputs[f"in_past_key_value.{idx}.key"] = key
        inputs[f"in_past_key_value.{idx}.value"] = value
    return inputs


def postprocess_text_decoder_outputs(output:Dict, past_key_values_outs:List[str]):
    """
    Helper function for rearranging model outputs and wrapping to CausalLMOutputWithCrossAttentions
    Parameters:
      output (Dict): dictionary with model output
    Returns
      wrapped_outputs (CausalLMOutputWithCrossAttentions): outputs wrapped to CausalLMOutputWithCrossAttentions format
    """
    outs = {k.any_name: v for k, v in output.items()}
    logits = torch.from_numpy(outs["logits"])
    past_kv = []
    for i in range(0, len(past_key_values_outs), 2):
        key = past_key_values_outs[i]
        value = key.replace(".key", ".value")
        past_kv.append((torch.from_numpy(outs[key]), torch.from_numpy(outs[value])))
    return CausalLMOutputWithCrossAttentions(
        loss=None,
        logits=logits,
        past_key_values=past_kv,
        hidden_states=None,
        attentions=None,
        cross_attentions=None
    )


def text_decoder_forward(
    input_ids:torch.Tensor,
    attention_mask:torch.Tensor,
    past_key_values:List[Tuple[torch.Tensor, torch.Tensor]],
    encoder_hidden_states:torch.Tensor,
    encoder_attention_mask:torch.Tensor,
    past_key_values_outs:List[str],
    ov_text_decoder:Model,
    ov_text_decoder_with_past:Model,
    **kwargs):
    """
    Inference function for text_decoder in one generation step
    Parameters:
      input_ids (torch.Tensor): input token ids
      attention_mask (torch.Tensor): attention mask for input token ids
      past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): list of cached decoder hidden states from previous step
      encoder_hidden_states (torch.Tensor): encoder (vision or text) hidden states
      encoder_attention_mask (torch.Tensor): attnetion mask for encoder hidden states
    Returns
      model outputs (CausalLMOutputWithCrossAttentions): model prediction wrapped to CausalLMOutputWithCrossAttentions class including predicted logits and hidden states for caching
    """
    input_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask
    }
    if past_key_values is None:
        outputs = ov_text_decoder(input_dict)
    else:
        input_dict.update(prepare_past_inputs(past_key_values))
        outputs = ov_text_decoder_with_past(input_dict)
    return postprocess_text_decoder_outputs(outputs, past_key_values_outs)


class OVBlipModel:
    """
    Model class for inference BLIP model with OpenVINO
    """
    def __init__(self, config, decoder_start_token_id:int, vision_model, text_encoder, text_decoder):
        """
        Initialization class parameters
        """
        self.vision_model = vision_model
        self.vision_model_out = vision_model.output(0)
        self.text_encoder = text_encoder
        self.text_encoder_out = text_encoder.output(0)
        self.text_decoder = text_decoder
        self.config = config
        self.decoder_start_token_id = decoder_start_token_id
        self.decoder_input_ids = config.text_config.bos_token_id

    def generate_answer(self, pixel_values:torch.Tensor, input_ids:torch.Tensor, attention_mask:torch.Tensor, **generate_kwargs):
        """
        Visual Question Answering prediction
        Parameters:
          pixel_values (torch.Tensor): preprocessed image pixel values
          input_ids (torch.Tensor): question token ids after tokenization
          attention_mask (torch.Tensor): attention mask for question tokens
        Retruns:
          generation output (torch.Tensor): tensor which represents sequence of generated answer token ids
        """
        image_embed = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]
        image_attention_mask = np.ones(image_embed.shape[:-1], dtype=int)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        question_embeds = self.text_encoder([input_ids.detach().numpy(), attention_mask.detach().numpy(), image_embed, image_attention_mask])[self.text_encoder_out]
        question_attention_mask = np.ones(question_embeds.shape[:-1], dtype=int)

        bos_ids = np.full((question_embeds.shape[0], 1), fill_value=self.decoder_start_token_id)

        outputs = self.text_decoder.generate(
            input_ids=torch.from_numpy(bos_ids),
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=torch.from_numpy(question_embeds),
            encoder_attention_mask=torch.from_numpy(question_attention_mask),
            **generate_kwargs,
        )
        return outputs

    def generate_caption(self, pixel_values:torch.Tensor, input_ids:torch.Tensor = None, attention_mask:torch.Tensor = None, **generate_kwargs):
        """
        Image Captioning prediction
        Parameters:
          pixel_values (torch.Tensor): preprocessed image pixel values
          input_ids (torch.Tensor, *optional*, None): pregenerated caption token ids after tokenization, if provided caption generation continue provided text
          attention_mask (torch.Tensor): attention mask for caption tokens, used only if input_ids provided
        Retruns:
          generation output (torch.Tensor): tensor which represents sequence of generated caption token ids
        """
        batch_size = pixel_values.shape[0]

        image_embeds = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]

        image_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
            )
        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=torch.from_numpy(image_embeds),
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs
