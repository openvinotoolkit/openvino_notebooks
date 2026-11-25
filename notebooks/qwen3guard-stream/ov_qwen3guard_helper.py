from pathlib import Path


def create_tiny_models() -> tuple[str, str]:
    """Create architecture-preserving random fixtures without downloading pretrained weights."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    source = "Qwen/Qwen3Guard-Stream-0.6B"
    revision = "74e1479150e9029d6778993f00491108323bb6f8"
    guard_dir = Path("models") / "tiny-qwen3guard-stream"
    generator_dir = Path("models") / "tiny-qwen3-generator"
    if all((directory / filename).is_file() for directory in (guard_dir, generator_dir) for filename in ("config.json", "model.safetensors", "tokenizer.json")):
        return str(guard_dir), str(generator_dir)

    config = AutoConfig.from_pretrained(source, revision=revision, trust_remote_code=True)
    config.hidden_size = 32
    config.intermediate_size = 128
    config.guard_inner_size = 32
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.max_window_layers = 2
    config.layer_types = ["full_attention"] * 2
    config.torch_dtype = "float32"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        guard_model = AutoModel.from_config(config, trust_remote_code=True, code_revision=revision, attn_implementation="eager").float().eval()
        guard_model.__class__.register_for_auto_class("AutoModel")
        guard_model.save_pretrained(guard_dir)
        generator_config = Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            tie_word_embeddings=True,
        )
        generator = AutoModelForCausalLM.from_config(generator_config).float().eval()
        generator.save_pretrained(generator_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    tokenizer.save_pretrained(guard_dir)
    tokenizer.save_pretrained(generator_dir)
    return str(guard_dir), str(generator_dir)
