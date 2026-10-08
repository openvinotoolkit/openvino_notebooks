"""Self-contained PyTorch reference for VLA-JEPA, used to validate the OpenVINO port.

The upstream project generates its reference tensors with ``run_baseline.py``, which
imports the VLA-JEPA training tree (``starVLA``) and additionally needs the V-JEPA 2
encoder. Neither is required at inference time, so reproducing that here would make the
notebook depend on a large source tree it never actually runs.

Instead this module rebuilds the *inference* path from the checkpoint using only
``transformers`` plus the action-head re-implementation in ``export.py`` (the same code
that gets converted), runs it in PyTorch, and returns those tensors as the reference.
That is the identical construction the IR is exported from, so comparing the two
measures exactly what we want to measure: the cost of the conversion.

Everything here resolves paths relative to the caller — no repository layout is assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from export import FlowmatchingActionHeadSingleStep, cos_sim, load_action_cfg  # noqa: F401

# The tokenizer ids VLA-JEPA was trained with. `export.py` re-adds the tokens in this
# order and asserts the resulting ids, so they are fixed rather than discovered.
FIRST_ACTION_TOKEN_ID = 151669
NUM_ACTION_TOKENS = 28  # action_horizon * 4
EMBODIED_ACTION_TOKEN_ID = 151697

# The training prompt reserves 8 copies each of <|action_0|>, <|action_1|>, <|action_2|>.
ACTION_TOKEN_GROUPS = 3


def build_config(config_yaml: Path, dataset_statistics: Path, instruction: str, unnorm_key: str | None = None, seed: int = 42) -> dict:
    """Derive the pipeline config that ``export.py`` / ``run_inference_standalone.py`` read.

    Upstream this file is a by-product of the PyTorch baseline run. Every field is in
    fact already determined by the training config and the dataset statistics shipped
    with the checkpoint, so we can build it directly and skip that dependency.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(config_yaml), resolve=True)
    fw = raw["framework"]
    am = fw["action_model"]
    vj2 = fw["vj2_model"]
    data = raw["datasets"]["vla_data"]

    stats = json.loads(Path(dataset_statistics).read_text())
    if unnorm_key is None:
        if len(stats) != 1:
            raise ValueError(f"{dataset_statistics} holds {len(stats)} datasets {list(stats)} — " "pass unnorm_key to choose one")
        unnorm_key = next(iter(stats))
    action_stats = stats[unnorm_key]["action"]

    n_steps = int(am["num_inference_timesteps"])
    buckets = int(am["num_timestep_buckets"])
    n_per_group = int(vj2["num_action_tokens_per_timestep"])
    action_tok = str(vj2["special_action_token"])
    embodied_tok = str(vj2["embodied_action_token"])
    n_embodied = int(vj2["num_embodied_action_tokens_per_instruction"])
    res = int(data["resolution_size"])

    return {
        "action_dim": int(am["action_dim"]),
        "action_horizon": int(am["action_horizon"]),
        "state_dim": int(am["state_dim"]),
        "num_inference_timesteps": n_steps,
        "num_timestep_buckets": buckets,
        "timesteps": [int(t / n_steps * buckets) for t in range(n_steps)],
        "dt": 1.0 / n_steps,
        "qwen_hidden_size": int(fw["qwenvl"]["vl_hidden_dim"]),
        "num_embodied_action_tokens": n_embodied,
        "embodied_action_token": embodied_tok,
        "embodied_action_token_id": EMBODIED_ACTION_TOKEN_ID,
        "action_token_ids": [FIRST_ACTION_TOKEN_ID + i for i in range(NUM_ACTION_TOKENS)],
        "dit_input_embedding_dim": 768,
        "dit_hidden_size": int(am["hidden_size"]),
        "dit_num_layers": int(am["diffusion_model_cfg"]["num_layers"]),
        "dit_cross_attention_dim": int(am["diffusion_model_cfg"]["cross_attention_dim"]),
        "dit_norm_type": str(am["diffusion_model_cfg"]["norm_type"]),
        "dit_interleave_self_attention": bool(am["diffusion_model_cfg"]["interleave_self_attention"]),
        "action_model_type": str(am["action_model_type"]),
        "cot_prompt": str(data["CoT_prompt"]),
        "action_prompt": "".join(action_tok.format(i) * n_per_group for i in range(ACTION_TOKEN_GROUPS)),
        "embodied_prompt": embodied_tok * n_embodied,
        "image_size": [res, res],
        "instruction": instruction,
        "rng_seed": seed,
        "unnorm_key": unnorm_key,
        "action_norm_stats": action_stats,
    }


def make_fixed_input(cfg: dict):
    """Deterministic LIBERO-style observation: two RGB views plus an 8-D state vector.

    Reproduces the baseline's fixed sample exactly (same seed, same draw order), so the
    reference tensors this module produces are stable across machines and runs.
    """
    rng = np.random.default_rng(cfg["rng_seed"])
    h, w = cfg["image_size"]
    primary = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # LIBERO state = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    state = rng.uniform(-1.0, 1.0, size=(1, cfg["state_dim"])).astype(np.float32)
    images = [Image.fromarray(primary), Image.fromarray(wrist)]
    return images, np.stack([primary, wrist]), state


def build_prompt(cfg: dict, instruction: str) -> str:
    return cfg["cot_prompt"].replace("{instruction}", instruction).replace("{actions}", cfg["action_prompt"]).replace("{e_actions}", cfg["embodied_prompt"])


def load_processor(qwen_base: Path):
    """Load the base processor and re-add VLA-JEPA's tokens in training order.

    Without this the string ``<|embodied_action|>`` tokenises into ordinary subwords, no
    position matches the expected id, and nothing is extracted.
    """
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(qwen_base)
    tok = proc.tokenizer
    new = [f"<|action_{i}|>" for i in range(NUM_ACTION_TOKENS)] + ["<|embodied_action|>"]
    tok.add_tokens([t for t in new if t not in tok.get_vocab()], special_tokens=True)
    got = tok.convert_tokens_to_ids("<|embodied_action|>")
    if got != EMBODIED_ACTION_TOKEN_ID:
        raise RuntimeError(f"<|embodied_action|> got id {got}, expected {EMBODIED_ACTION_TOKEN_ID} — " "token order does not match the training tokenizer")
    return proc


def preprocess(processor, cfg: dict, images: Sequence[Image.Image], instruction: str):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": build_prompt(cfg, instruction)})
    return processor.apply_chat_template(
        [[{"role": "user", "content": content}]],
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def load_qwen_reference(checkpoint: Path, qwen_base: Path):
    """Rebuild the backbone exactly as ``export.py`` converts it.

    Two things matter and are easy to get wrong:

    * weights are loaded ``strict``-ly — a lenient load silently leaves a randomly
      initialised model that produces perfectly shaped garbage;
    * the text model's final RMSNorm is replaced by an identity, because VLA-JEPA
      consumes the *pre*-norm hidden state and the norm is not invertible downstream.
    """
    from transformers import AutoConfig, Qwen3VLModel

    hf_cfg = AutoConfig.from_pretrained(qwen_base)
    hf_cfg.torch_dtype = "float32"
    model = Qwen3VLModel(hf_cfg)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "qwen_vl_interface.model.model."
    sd = {k[len(prefix) :]: v.float() for k, v in ckpt.items() if k.startswith(prefix)}
    if not sd:
        raise RuntimeError(f"no keys with prefix {prefix!r} in {checkpoint}")
    del ckpt

    rows = sd["language_model.embed_tokens.weight"].shape[0]
    if rows != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(rows)
    model.load_state_dict(sd, strict=True)
    del sd

    model.language_model.norm = torch.nn.Identity()
    return model.eval().float()


def load_dit_reference(checkpoint: Path, config_yaml: Path):
    """Rebuild the action head from the same class ``export.py`` converts."""
    head = FlowmatchingActionHeadSingleStep(load_action_cfg(Path(config_yaml))).eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = {k[len("action_model.") :]: v.float() for k, v in ckpt.items() if k.startswith("action_model.")}
    head.load_state_dict(sd, strict=True)
    del ckpt, sd
    return head


def extract_embodied_tokens(hidden: np.ndarray, input_ids: np.ndarray, cfg: dict) -> np.ndarray:
    rows, cols = np.nonzero(input_ids == cfg["embodied_action_token_id"])
    if rows.size == 0:
        raise RuntimeError("no <|embodied_action|> position matched — prompt/tokenizer mismatch")
    tokens = hidden[rows, cols, :].reshape(input_ids.shape[0], -1, hidden.shape[-1])
    if tokens.shape[1] != cfg["num_embodied_action_tokens"]:
        raise RuntimeError(f"extracted {tokens.shape[1]} embodied tokens, " f"expected {cfg['num_embodied_action_tokens']}")
    return tokens


def generate_reference(out_dir: Path, cfg: dict, checkpoint: Path, config_yaml: Path, qwen_base: Path) -> dict:
    """Run the full PyTorch pipeline on the fixed observation and save the tensors.

    Writes the same file names the OpenVINO scripts expect, so the notebook's reference
    directory is a drop-in for the upstream ``golden_outputs/``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, images_np, state = make_fixed_input(cfg)
    instruction = cfg["instruction"]

    processor = load_processor(qwen_base)
    inputs = preprocess(processor, cfg, images, instruction)

    print("  running Qwen3-VL backbone in PyTorch (fp32, CPU) …")
    qwen = load_qwen_reference(checkpoint, qwen_base)
    with torch.no_grad():
        hidden = qwen(**inputs).last_hidden_state.float().numpy()
    del qwen

    input_ids = np.asarray(inputs["input_ids"])
    embodied = extract_embodied_tokens(hidden, input_ids, cfg)

    print("  running DiT action head in PyTorch …")
    head = load_dit_reference(checkpoint, config_yaml)

    n_steps = cfg["num_inference_timesteps"]
    buckets = cfg["num_timestep_buckets"]
    dt = 1.0 / n_steps
    B = embodied.shape[0]

    # A fixed noise draw keeps the whole reference reproducible; in deployment this is
    # sampled fresh every call.
    rng = np.random.default_rng(cfg["rng_seed"])
    actions = rng.standard_normal((B, cfg["action_horizon"], cfg["action_dim"]), dtype=np.float32)
    initial_noise = actions.copy()

    emb_t = torch.from_numpy(np.ascontiguousarray(embodied, dtype=np.float32))
    state_t = torch.from_numpy(np.asarray(state, np.float32).reshape(B, 1, cfg["state_dim"]))

    noisy_per_step, velocity_per_step = [], []
    with torch.no_grad():
        for t in range(n_steps):
            t_disc = int(t / n_steps * buckets)
            noisy_per_step.append(actions.copy())
            velocity = head(
                torch.from_numpy(np.ascontiguousarray(actions)),
                torch.tensor([t_disc] * B, dtype=torch.int64),
                emb_t,
                state_t,
            ).numpy()
            velocity_per_step.append(velocity)
            actions = actions + dt * velocity
    del head

    from run_inference_standalone import unnormalize_actions

    unnormalized = unnormalize_actions(actions[0].copy(), cfg["action_norm_stats"])

    tensors = {
        "input_images": images_np,
        "input_state": state,
        "embodied_action_tokens": embodied,
        "initial_noise": initial_noise,
        "pred_actions": actions,
        "unnormalized_actions": unnormalized,
    }
    for i in range(n_steps):
        tensors[f"dit_noisy_actions_step{i}"] = noisy_per_step[i]
        tensors[f"dit_velocity_step{i}"] = velocity_per_step[i]

    for name, arr in tensors.items():
        np.save(out_dir / f"{name}.npy", np.asarray(arr))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"  wrote {len(tensors)} tensors + config.json to {out_dir}")
    return tensors
