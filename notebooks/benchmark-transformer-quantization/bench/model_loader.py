import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM

class ModelFactory:
    @staticmethod
    def load_pytorch(model_id):
        print(f"Loading PyTorch Model: {model_id}...")
        start = time.perf_counter()
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        load_time = time.perf_counter() - start
        
        # Return 4 values to match notebook expectations (Metric is None for PyTorch)
        return model, tokenizer, load_time, None

    @staticmethod
    def load_openvino(model_id, precision="int4", device="CPU", cache_dir="./ov_cache"):
        """
        Loads an OpenVINO model with device selection.
        """
        print(f" Loading OpenVINO Model: {model_id} on {device}...")
        start = time.perf_counter()
        
        # 1. Determine Cache Path vs Direct Download
        if precision == "already_quantized":
            # FAST PATH: Load pre-optimized model directly from HF
            model = OVModelForCausalLM.from_pretrained(
                model_id,
                device=device  # <--- UPDATED: Uses the widget selection
            )
            cache_path = Path(cache_dir) / "pre_quantized_download"
        else:
            # SLOW PATH: Local Export & Compression
            cache_path = Path(cache_dir) / f"{model_id.replace('/', '_')}_{precision}"
            
            export_config = {"trust_remote_code": True}
            if precision == "int4":
                export_config["load_in_4bit"] = True
                export_config["quantization_config"] = {"bits": 4, "sym": True, "group_size": 128}
            
            if not cache_path.exists():
                print("   ↳ Exporting to IR (this may take time)...")
                model = OVModelForCausalLM.from_pretrained(
                    model_id, 
                    export=True,
                    device=device,  # UPDATED: Uses the widget selection
                    **export_config
                )
                model.save_pretrained(cache_path)
            else:
                model = OVModelForCausalLM.from_pretrained(
                    cache_path,
                    device=device   # UPDATED: Uses the widget selection
                )
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_time = time.perf_counter() - start
        
        # Return 4 values (Cache path is the metric here)
        return model, tokenizer, load_time, cache_path