from transformers import PretrainedConfig, AutoTokenizer

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any, List
from pathlib import Path
import openvino as ov
import torch
import numpy as np


class OVMPTModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for MPT
    """

    def __init__(
        self,
        model: "Model",
        config: "PretrainedConfig" = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf["mpt"] = NormalizedTextConfig.with_args(
            num_layers="n_layers", num_attention_heads="n_heads"
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )

    def _reshape(self, model: "Model", *args, **kwargs):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if shapes[inputs].rank.get_length() in [2, 3]:
                shapes[inputs][1] = -1
            else:
                if ".key" in inputs.get_any_name():
                    shapes[inputs][3] = -1
                else:
                    shapes[inputs][2] = -1

        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads
                if self.config.model_type == "bloom"
                else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = ov.Tensor(
                    model_inputs.get_element_type(), shape.get_shape()
                )

        inputs["input_ids"] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(
            self.device
        )

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv]
                for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVMPTModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


class OVQWENModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for QWEN
    """

    def __init__(
        self,
        model: "Model",
        config: "PretrainedConfig" = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf["qwen"] = NormalizedTextConfig.with_args(
            num_layers="num_hidden_layers",
            num_attention_heads="num_attention_heads",
            hidden_size="hidden_size",
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )

    def _reshape(self, model: "Model", *args, **kwargs):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads
                if self.config.model_type == "bloom"
                else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = ov.Tensor(
                    model_inputs.get_element_type(), shape.get_shape()
                )

        inputs["input_ids"] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = np.array(attention_mask)
        if "token_type_ids" in self.input_names and attention_mask is not None:
            inputs["token_type_ids"] = np.array(attention_mask)
        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(
            self.device
        )

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv]
                for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVQWENModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )

class OVCHATGLMModel(OVModelForCausalLM):
    """
    Optimum intel compatible model wrapper for CHATGLM2
    """

    def __init__(
        self,
        model: "Model",
        config: "PretrainedConfig" = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        NormalizedConfigManager._conf["chatglm"] = NormalizedTextConfig.with_args(
            num_layers="num_hidden_layers",
            num_attention_heads="num_attention_heads",
            hidden_size="hidden_size",
        )
        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs
        )
    
    def _reshape(self, model: "Model", *args, **kwargs):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if "past_key_values" in inputs.get_any_name():
                shapes[inputs][0] = -1
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads
                if self.config.model_type == "bloom"
                else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[1] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[0].is_dynamic:
                    shape[0] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = ov.Tensor(
                    model_inputs.get_element_type(), shape.get_shape()
                )
            

        inputs["input_ids"] = np.array(input_ids)

        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = np.array(attention_mask)
        if "position_ids" in self.input_names and attention_mask is not None:
            inputs["position_ids"] = np.array(position_ids)
        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(
            self.device
        )

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv]
                for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None
            
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        init_cls = OVCHATGLMModel

        return init_cls(
            model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs
        )


model_classes = {
    "mpt": OVMPTModel,
    "qwen": OVQWENModel,
    "chatglm3": OVCHATGLMModel,
}