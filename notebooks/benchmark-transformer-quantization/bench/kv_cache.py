import gc
import torch

class KVCacheManager:
    """
    Helper to manage memory state between benchmark runs.
    """
    @staticmethod
    def clear():
        """
        Aggressively clears PyTorch and Python garbage to prevent 
        OOM (Out of Memory) during model swapping.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @staticmethod
    def validate_config(model_config):
        """
        Ensures the model config has use_cache=True.
        """
        if hasattr(model_config, "use_cache") and not model_config.use_cache:
            print("WARNING: Model 'use_cache' is False. Benchmarking generation without KV-cache is slow.")