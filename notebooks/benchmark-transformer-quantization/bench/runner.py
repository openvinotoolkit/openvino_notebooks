import torch
import time
import numpy as np
from tqdm import tqdm
from .inputs import TransformerInputGenerator
from .metrics import MetricsCollector
from .kv_cache import KVCacheManager

class BenchmarkRunner:
    def __init__(self, model, tokenizer, framework="pt"):
        self.model = model
        self.generator = TransformerInputGenerator(tokenizer)
        self.framework = framework
        self.metrics = MetricsCollector()
        
        # Validate config on startup
        if hasattr(model, "config"):
            KVCacheManager.validate_config(model.config)

    def run(self, num_iters=50, warmup=5):
        # Clear memory before starting
        KVCacheManager.clear()
        
        inputs = self.generator.get_inputs()
        
        print(f" Starting Benchmark [{self.framework}]...")
        
        # 1. Warmup
        for _ in range(warmup):
            with torch.no_grad():
                self.model(**inputs)

        # 2. Measurement Loop
        latencies = []
        start_global = time.perf_counter()
        
        # tqdm progress bar
        for _ in tqdm(range(num_iters), desc=f"{self.framework} Bench"):
            t0 = time.perf_counter()
            with torch.no_grad():
                self.model(**inputs)
            latencies.append((time.perf_counter() - t0) * 1000) # ms

        duration = time.perf_counter() - start_global
        
        # 3. CRITICAL: THIS RETURN BLOCK MUST EXIST!
        return {
            "p50_ms": np.median(latencies),
            "p99_ms": np.percentile(latencies, 99),
            "throughput_ips": num_iters / duration,
            "ram_usage_mb": self.metrics.get_current_rss_mb()
        }