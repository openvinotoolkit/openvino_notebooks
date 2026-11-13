"""Helpers for the LingBot-VLA v2 OpenVINO notebook.

The model code is the upstream LingBot-VLA v2 repository at a pinned commit. Upstream targets CUDA
(flash-attn, Triton grouped-GEMM MoE kernels, `.cuda()` calls), so `LingBotPolicy` patches a few
entry points at runtime so that the policy loads and runs in FP32 on CPU. `convert_to_openvino`
then replaces the data-dependent parts of `sample_actions` with traceable equivalents, checks them
against the PyTorch model and converts the whole policy, including the flow-matching loop, into one
OpenVINO model.
"""

import contextlib
import gc
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM_URL = "https://github.com/Robbyant/lingbot-vla-v2.git"
UPSTREAM_COMMIT = "be969b8fd117fb70550c5d4bf4bc328211b5b1b6"

INPUT_NAMES = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "noise")
CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")


def clone_upstream(repo_dir):
    """Clone the upstream repository and check out the pinned commit (no-op if already there)."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "clone", "-q", UPSTREAM_URL, str(repo_dir)], check=True)
    head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", UPSTREAM_COMMIT], check=True)
    return repo_dir


@contextlib.contextmanager
def _in_upstream(repo_dir):
    """Upstream resolves robot configs and normalization statistics relative to the repository root."""
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        yield
    finally:
        os.chdir(cwd)


def _import_upstream():
    with contextlib.ExitStack() as stack:
        if not torch.cuda.is_available():
            # lingbotvla.ops picks pre-tuned Triton configs by CUDA device name at import time.
            stack.enter_context(mock.patch("torch.cuda.get_device_capability", return_value=(0, 0)))
            stack.enter_context(mock.patch("torch.cuda.get_device_name", return_value="cpu"))
        import deploy.lingbot_vla_v2_policy as policy
        from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla_v2, qwen2_action_expert
    return policy, modeling_lingbot_vla_v2, qwen2_action_expert


def _sdpa_or_flash(check):
    """Upstream hardcodes flash_attention_2; fall back to SDPA when flash-attn is not installed."""

    def wrapper(self, attn_implementation, *args, **kwargs):
        if attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            attn_implementation = "sdpa"
        return check(self, attn_implementation, *args, **kwargs)

    return wrapper


def _routed_experts_forward(self, module, num_experts, routing_weights, selected_experts, hidden_states):
    """Replacement for the Triton grouped-GEMM forward of Qwen2FusedExperts.

    The reference path runs each expert on the tokens routed to it. After `_use_dense_experts`, the
    export path runs every expert on every token as two wide FCs (gate/up, E * F outputs), scales
    the activations by the dense routing weights and applies one down FC with K = E * F, which
    also sums the expert outputs. Both compute sum_k w_k * expert_k(x) over the top-k experts.
    """
    if getattr(self, "dense_weights", None) is not None:
        gate, up, down = self.dense_weights
        tokens = hidden_states.shape[0]
        weights = torch.zeros(tokens, num_experts, dtype=torch.float32).scatter(1, selected_experts, routing_weights.float())
        scale = weights.to(hidden_states.dtype).unsqueeze(-1).expand(tokens, num_experts, self.intermediate_size)
        h = F.silu(F.linear(hidden_states, gate)) * F.linear(hidden_states, up)
        return torch.matmul(h * scale.reshape(tokens, -1), down)

    out = torch.zeros_like(hidden_states)
    for expert in range(num_experts):
        token_idx, slot = torch.where(selected_experts == expert)
        if token_idx.numel() == 0:
            continue
        x = hidden_states[token_idx]
        h = F.silu(F.linear(x, self.gate_proj[expert])) * F.linear(x, self.up_proj[expert])
        out.index_add_(0, token_idx, F.linear(h * routing_weights[token_idx, slot, None], self.down_proj[expert]))
    return out


def _use_dense_experts(model, experts_cls):
    blocks = [m for m in model.modules() if isinstance(m, experts_cls)]
    for m in blocks:
        num_experts, inter, dim = m.gate_proj.shape
        # The down weight is stored as (E * F, D) and applied with matmul: OpenVINO GPU then gives every
        # denoising step its own copy, which avoids sharing one transposed FC weight with K >= 8192.
        down = m.down_proj.detach().permute(0, 2, 1).reshape(num_experts * inter, dim).contiguous()
        m.dense_weights = (m.gate_proj.detach().reshape(-1, dim), m.up_proj.detach().reshape(-1, dim), down)
    return len(blocks)


def _load_weights(vla, ckpt_dir):
    from safetensors.torch import load_file

    expected, loaded = set(vla.state_dict()), set()
    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    for i, path in enumerate(files):
        state = load_file(path)
        result = vla.load_state_dict(state, strict=False, assign=True)
        if result.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint tensors: {result.unexpected_keys[:5]}")
        loaded.update(state)
        del state
        gc.collect()
        print(f"loaded {path.name} ({i + 1}/{len(files)})", flush=True)
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"checkpoint is missing tensors: {sorted(missing)[:5]}")


class _RawImageProcessor:
    """Stands in for the Qwen image processor: keeps the resized CHW image and reports its grid.

    Tokenization is unchanged, and rescale/normalize/patchify move into the converted model.
    """

    def __init__(self, image_processor):
        self._ip = image_processor

    def __getattr__(self, name):
        return getattr(self._ip, name)

    def __call__(self, img):
        factor = self._ip.patch_size * self._ip.merge_size
        h, w = img.shape[-2:]
        if h % factor or w % factor:
            raise ValueError(f"image size {h}x{w} must be a multiple of {factor}")
        grid = torch.tensor([[1, h // self._ip.patch_size, w // self._ip.patch_size]], dtype=torch.long)
        return {"pixel_values": img, "image_grid_thw": grid}


def qwen_preprocess(images, img_masks, patch_size=16, merge_size=2, temporal_patch_size=2):
    """(B, N, C, H, W) images in [0, 255] -> (B, N, L, C*T*P*P) Qwen3-VL patches.

    Same as the Qwen2-VL image processor without resizing (mean = std = 0.5); missing cameras become
    -1.0 like the padding in prepare_images.
    """
    b, n, c, h, w = images.shape
    x = torch.where(img_masks[:, :, None, None, None], images, torch.zeros_like(images)) / 127.5 - 1.0
    gh, gw = h // patch_size, w // patch_size
    x = x.reshape(b * n, c, gh // merge_size, merge_size, patch_size, gw // merge_size, merge_size, patch_size)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.reshape(b * n * gh * gw, c, 1, patch_size * patch_size).expand(-1, -1, temporal_patch_size, -1)
    return x.reshape(b, n, gh * gw, c * temporal_patch_size * patch_size * patch_size)


class LingBotPolicy:
    """Upstream LingbotVLAv2Server (preprocessing, normalization) plus the PyTorch model on CPU.

    With load_weights=False the model is built on the meta device: preprocessing and
    postprocessing work, the PyTorch model does not.
    """

    def __init__(self, repo_dir, ckpt_dir, qwen_dir, robo_name="robotwin", load_weights=True):
        self.repo_dir = Path(repo_dir).resolve()
        ckpt_dir = Path(os.path.abspath(ckpt_dir))
        os.environ["QWEN3VL_PATH"] = str(Path(qwen_dir).resolve())
        with _in_upstream(repo_dir):
            policy, self.v2mod, expert_mod = _import_upstream()
            from transformers import PreTrainedModel
            from transformers.modeling_utils import no_init_weights

            self._experts_cls = expert_mod.Qwen2FusedExperts
            self._experts_cls.forward = _routed_experts_forward

            policy_cls = policy.LingBotVlaV2InferencePolicy

            def build(config, eval=True):
                with torch.device("meta") if not load_weights else no_init_weights():
                    return policy_cls(config, eval=eval)

            class Server(policy.LingbotVLAv2Server):
                def load_model_weights(self, path_to_pi_model, strict=True):
                    if load_weights:
                        _load_weights(self.vla, path_to_pi_model)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(policy, "LingBotVlaV2InferencePolicy", build))
                stack.enter_context(
                    mock.patch.object(
                        PreTrainedModel, "_check_and_adjust_attn_implementation", _sdpa_or_flash(PreTrainedModel._check_and_adjust_attn_implementation)
                    )
                )
                # upstream moves the policy to CUDA unconditionally
                stack.enter_context(mock.patch.object(torch.nn.Module, "cuda", lambda module, *args, **kwargs: module))
                self.server = Server(str(ckpt_dir), use_length=50, chunk_ret=True, use_bf16=False, use_fp32=True)
                self.server.infer(dict(reset=True, robo_name=robo_name))
        self.server.processor.image_processor = _RawImageProcessor(self.server.processor.image_processor)
        self.model = self.server.vla.model
        self.model.requires_grad_(False)
        self.config = self.model.config

    def preprocess(self, observation):
        """Raw observation -> dict of numpy model inputs (images stay resized CHW in [0, 255])."""
        # Same steps as LingbotVLAv2Server._infer_batch, minus the CUDA-only model call.
        with _in_upstream(self.repo_dir):
            applied = self.server._prepare_model_input(dict(observation))
        batch = {k: self.server._pad_and_stack_tensors([v]) for k, v in applied.items() if isinstance(v, torch.Tensor)}
        inputs = {k: batch[k].float().numpy() if k in ("images", "state") else batch[k].numpy() for k in INPUT_NAMES[:-1]}
        inputs["image_grid_thw"] = batch["image_grid_thw"].numpy()
        return inputs

    def postprocess(self, observation, actions):
        """Normalized (1, 50, 55) model output -> unnormalized robot actions."""
        with _in_upstream(self.repo_dir):
            applied = [self.server._prepare_model_input(dict(observation))]
            chunk = self.server._unapply_batched_actions(applied, torch.as_tensor(np.asarray(actions)))
        return {k: v[0] for k, v in chunk.items()}

    @torch.no_grad()
    def sample_actions(self, inputs, noise):
        """PyTorch FP32 reference: the upstream FlowMatchingV2.sample_actions."""
        t = {k: torch.from_numpy(np.asarray(v)) for k, v in inputs.items()}
        ip = self.server.processor.image_processor
        patches = qwen_preprocess(t["images"], t["img_masks"], ip.patch_size, ip.merge_size, ip.temporal_patch_size)
        return self.model.sample_actions(
            patches, t["img_masks"], t["lang_tokens"], t["lang_masks"], t["state"], noise=torch.from_numpy(noise).clone(), image_grid_thw=t["image_grid_thw"]
        ).numpy()


ROBOTWIN_REPO = "TianxingChen/RoboTwin2.0"
ROBOTWIN_CAMERAS = ("head_camera", "left_camera", "right_camera")


def download_robotwin_episode(local_dir, task="lift_pot", episode=0, config="aloha-agilex_clean_50"):
    """Fetch one recorded episode (HDF5 + instructions, ~8 MB) from a RoboTwin 2.0 task archive.

    Every task is stored as one zip archive of 50 episodes; only the members of a single episode are
    read from it, so the whole archive is not downloaded.
    """
    import zipfile

    from huggingface_hub import HfFileSystem

    local_dir = Path(local_dir)
    members = [f"{config}/data/episode{episode}.hdf5", f"{config}/instructions/episode{episode}.json"]
    if not all((local_dir / m).exists() for m in members):
        archive = f"datasets/{ROBOTWIN_REPO}/dataset/{task}/{config}.zip"
        with HfFileSystem().open(archive, block_size=1 << 22) as f, zipfile.ZipFile(f) as zf:
            for m in members:
                zf.extract(m, local_dir)
    return [local_dir / m for m in members]


class RoboTwinEpisode:
    """A recorded RoboTwin 2.0 episode: JPEG camera streams, 14-D joint trajectory and instructions."""

    def __init__(self, hdf5_path, instructions_path):
        import h5py

        with h5py.File(hdf5_path, "r") as f:
            self.jpegs = {cam: f[f"observation/{cam}/rgb"][:] for cam in ROBOTWIN_CAMERAS}
            self.joints = f["joint_action/vector"][:].astype(np.float32)
        self.instructions = json.loads(Path(instructions_path).read_text())["seen"]

    def __len__(self):
        return len(self.joints)

    def images(self, t):
        from PIL import Image

        return [np.array(Image.open(io.BytesIO(self.jpegs[cam][t])).convert("RGB")) for cam in ROBOTWIN_CAMERAS]

    def observation(self, t, instruction=None):
        """Frame `t` in the format of the LingBot RoboTwin policy server."""
        obs = dict(zip(CAMERAS, self.images(t)))
        obs["observation.state"] = self.joints[t]
        obs["task"] = instruction or self.instructions[0]
        return obs


JOINT_NAMES = [f"{arm} {j}" for arm in ("left", "right") for j in (*(f"joint {i}" for i in range(1, 7)), "gripper")]


def render_replay_video(episode, chunks, video_path, joints=(1, 2, 6, 8, 9, 13), fps=10):
    """Write an H.264 video: camera views of every frame over the recorded and the predicted joint trajectories.

    `chunks` maps the frame index where the policy was queried to the predicted (50, 14) action chunk;
    the chunk predicted from frame `t` starts at frame `t + 1`.
    """
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = sorted(chunks)
    steps = np.arange(len(episode))
    fig = plt.figure(figsize=(12, 6.4), dpi=80)
    grid = fig.add_gridspec(2, len(joints), height_ratios=(1.25, 1))
    cam_axes = [fig.add_subplot(grid[0, i * len(joints) // 3 : (i + 1) * len(joints) // 3]) for i in range(3)]
    joint_axes = [fig.add_subplot(grid[1, i]) for i in range(len(joints))]
    title = fig.suptitle("")
    cam_images = [ax.imshow(img) for ax, img in zip(cam_axes, episode.images(0))]
    for ax, name in zip(cam_axes, ("head", "left wrist", "right wrist")):
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    pred_lines, cursors = [], []
    for ax, j in zip(joint_axes, joints):
        ax.plot(steps, episode.joints[:, j], color="tab:gray", lw=2, label="recorded")
        pred_lines.append(ax.plot([], [], color="tab:orange", lw=1.5, label="predicted")[0])
        cursors.append(ax.axvline(0, color="tab:red", lw=0.8))
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.set_xlim(0, len(episode) - 1)
        ax.tick_params(labelsize=7)
    joint_axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()

    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.width, stream.height = fig.canvas.get_width_height()
        for t in steps:
            start = max(s for s in starts if s <= t)
            chunk = chunks[start][: len(episode) - start - 1]
            for im, img in zip(cam_images, episode.images(t)):
                im.set_data(img)
            for line, cursor, j in zip(pred_lines, cursors, joints):
                line.set_data(np.arange(start + 1, start + 1 + len(chunk)), chunk[:, j])
                cursor.set_xdata([t, t])
            title.set_text(f"{episode.instructions[0]!r}   frame {t:3d}/{len(episode) - 1}   chunk from frame {start}")
            fig.canvas.draw()
            frame = av.VideoFrame.from_ndarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    plt.close(fig)
    return video_path


def _rope_index_vectorized(image_token_id, num_patch, llm_h, llm_w):
    def rope_index(input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        """Same result as Qwen3VLModel.get_rope_index for single-frame images, without python loops."""
        valid = attention_mask.bool()
        is_img = input_ids == image_token_id
        k = (torch.cumsum(is_img.long(), dim=1) - 1) % num_patch
        last_img = is_img & (k == num_patch - 1)
        adv = torch.where(is_img, last_img.long() * max(llm_h, llm_w), torch.ones_like(k)) * valid.long()
        start = torch.cumsum(adv, dim=1) - adv
        h_off = torch.where(is_img, k // llm_w, torch.zeros_like(k))
        w_off = torch.where(is_img, k % llm_w, torch.zeros_like(k))
        return torch.stack([start, start + h_off, start + w_off], dim=0) * valid.long().unsqueeze(0)

    return rope_index


def _sdpa_attention(query_states, key_states, value_states, attention_mask):
    """Action-expert attention via F.scaled_dot_product_attention, same contract as upstream's eager one.

    The finite mask fill keeps fully masked padding rows finite in FP16.
    """
    bsize, seq_len, num_heads, head_dim = query_states.shape
    b, kv_len, kv_heads, d = key_states.shape
    groups = num_heads // kv_heads
    if groups > 1:
        key_states = key_states[:, :, :, None].expand(b, kv_len, kv_heads, groups, d).reshape(b, kv_len, num_heads, d)
        value_states = value_states[:, :, :, None].expand(b, kv_len, kv_heads, groups, d).reshape(b, kv_len, num_heads, d)
    q, k, v = (t.transpose(1, 2) for t in (query_states, key_states, value_states))
    mask = torch.zeros(attention_mask.shape, dtype=q.dtype).masked_fill(~attention_mask, -30000.0)[:, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(bsize, seq_len, num_heads * head_dim)


def _batched_vision_attention(model, grid):
    """Qwen3-VL ViT attention as one SDPA over all (equally sized) camera images.

    Upstream runs one attention per image through cu_seqlens; with a fixed image grid the images
    batch, and the grid-derived rotary and position embeddings become FP32 constants.
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as mq

    n_img = int(grid.shape[0])

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = mq.apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        q, k, v = (t.reshape(n_img, seq // n_img, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        return self.proj(out.transpose(1, 2).reshape(seq, -1))

    blocks = [m for m in model.modules() if isinstance(m, mq.Qwen3VLVisionAttention)]
    for m in blocks:
        m.forward = forward.__get__(m)
    visual = model.qwenvl_with_expert.qwenvl.visual
    for name in ("preprcess_grid_thw", "fast_pos_embed_interpolate"):
        value = getattr(visual, name)(grid)
        setattr(visual, name, lambda grid_thw, value=value: value)
    return len(blocks)


def _check_rope(qwe, rope_index, vl_cfg, grid, num_images, num_patch, lang_len):
    """Compare the vectorized M-RoPE ids with get_rope_index for every language length and random missing cameras."""
    image_token_len = num_patch + 2
    torch.manual_seed(1)
    for n_lang in range(lang_len + 1):
        ids = torch.cat(
            [
                torch.tensor([vl_cfg.vision_start_token_id] + [vl_cfg.image_token_id] * num_patch + [vl_cfg.vision_end_token_id]).repeat(num_images),
                torch.full((lang_len,), 100),
                torch.full((16,), vl_cfg.text_config.eos_token_id),
            ]
        )[None]
        img_valid = torch.rand(num_images) < 0.7
        img_valid[0] = True
        mask = torch.cat([img_valid.repeat_interleave(image_token_len), torch.arange(lang_len) < n_lang, torch.ones(16, dtype=torch.bool)])[None].long()
        ref = qwe.build_prefix_position_ids(ids, mask, image_grid_thw=grid[: int(img_valid.sum())])
        new = rope_index(ids, mask)
        valid = mask.bool().unsqueeze(0).expand_as(ref)
        if ref.shape != new.shape or not bool((ref[valid] == new[valid]).all()):
            raise RuntimeError(f"vectorized M-RoPE ids differ from get_rope_index for language length {n_lang}")


class _Full(torch.nn.Module):
    """images, masks, language, state, noise -> actions, with the flow-matching loop unrolled."""

    def __init__(self, policy, inputs):
        super().__init__()
        self.model = policy.model  # registers the weights as parameters for tracing
        qwe = policy.model.qwenvl_with_expert
        vl_cfg = qwe.qwenvl.config
        ip = policy.server.processor.image_processor
        self.patch_args = (ip.patch_size, ip.merge_size, ip.temporal_patch_size)
        grid = torch.from_numpy(inputs["image_grid_thw"]).reshape(-1, 3)
        if not bool((grid == grid[0]).all()) or int(grid[0, 0]) != 1:
            raise ValueError("conversion needs one single-frame image grid for all cameras")
        self.grid = grid
        self.bsize, self.num_images = inputs["img_masks"].shape
        merge = qwe.qwenvl.visual.spatial_merge_size
        self.llm_h, self.llm_w = int(grid[0, 1]) // merge, int(grid[0, 2]) // merge
        self.num_patch = self.llm_h * self.llm_w
        self.num_layers = vl_cfg.text_config.num_hidden_layers
        self.rope_index = _rope_index_vectorized(vl_cfg.image_token_id, self.num_patch, self.llm_h, self.llm_w)
        _check_rope(qwe, self.rope_index, vl_cfg, grid, self.num_images, self.num_patch, inputs["lang_tokens"].shape[1])

        # same float32 time accumulation as sample_actions (time += dt while time >= -dt / 2)
        self.dt = np.float32(-1.0 / policy.config.num_steps)
        self.timesteps, tv = [], np.float32(1.0)
        while tv >= -self.dt / 2:
            self.timesteps.append(float(tv))
            tv = np.float32(tv + self.dt)

    def deepstack(self, ds_list, img_masks, prefix_len):
        """(B*N, P, D) deepstack features -> (B, prefix_len, D) tensors added to the image token positions."""
        out = []
        for ds in ds_list:
            d = ds.shape[-1]
            ds = ds.reshape(self.bsize, self.num_images, self.num_patch, d) * img_masks[:, :, None, None].to(ds.dtype)
            pad = torch.zeros(self.bsize, self.num_images, 1, d, dtype=ds.dtype)
            blk = torch.cat([pad, ds, pad], dim=2).reshape(self.bsize, -1, d)
            out.append(torch.cat([blk, torch.zeros(self.bsize, prefix_len - blk.shape[1], d, dtype=ds.dtype)], dim=1))
        return out

    def prefix(self, pixel_values, img_masks, lang_tokens, lang_masks):
        model = self.model
        qwe = model.qwenvl_with_expert
        img_emb, ds = qwe.embed_image(pixel_values, self.grid)
        qwe.embed_image = lambda image, thw: (img_emb, ds)
        qwe.build_prefix_position_ids = self.rope_index
        try:
            dummy = torch.zeros(self.bsize, self.num_images, 1, 1, dtype=img_emb.dtype)
            embs, pad_masks, att_masks, pos_ids, vis_masks, _ = model.embed_prefix(dummy, img_masks, lang_tokens, lang_masks, image_grid_thw=self.grid)
            ds_full = self.deepstack(ds, img_masks, embs.shape[1])
            qwe._apply_deepstack = lambda h, i, m, d: (h + ds_full[i].to(h.dtype)) if i < len(ds_full) else h
            # torch.cumsum promotes bool to int64, the OpenVINO frontend keeps bool: cast first
            att_2d = self.make_att_2d_masks(pad_masks, att_masks.long())
            _, pkv, _ = qwe.forward(
                attention_mask=att_2d,
                position_ids=pos_ids,
                vlm_position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
                fill_kv_cache=True,
                visual_pos_masks=vis_masks,
                deepstack_visual_embeds=None,
            )
        finally:
            for name in ("embed_image", "build_prefix_position_ids", "_apply_deepstack"):
                qwe.__dict__.pop(name, None)
        return pad_masks, pos_ids, pkv

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        pixel_values = qwen_preprocess(images, img_masks, *self.patch_args)
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
        pad_masks, pos_ids, pkv = self.prefix(pixel_values, img_masks, lang_tokens, lang_masks)
        x_t = noise
        for tv in self.timesteps:
            t = torch.full((self.bsize,), tv, dtype=torch.float32)
            x_t = x_t + float(self.dt) * self.model.predict_velocity(state, pad_masks, pkv, x_t, t, prefix_position_ids=pos_ids)
        return x_t


@torch.no_grad()
def convert_to_openvino(policy, inputs, noise, reference_actions, xml_path):
    """Trace the whole policy into one static-shape OpenVINO model and save it with FP16 weights."""
    import openvino as ov

    model, qwe = policy.model, policy.model.qwenvl_with_expert
    full = _Full(policy, inputs)
    full.make_att_2d_masks = policy.v2mod.make_att_2d_masks
    qwe.attention_interface = _sdpa_attention
    n_vit = _batched_vision_attention(model, full.grid)
    n_moe = _use_dense_experts(model, policy._experts_cls)
    print(f"batched ViT attention in {n_vit} blocks, SDPA action-expert attention, dense experts in {n_moe} MoE blocks")

    example = {k: torch.from_numpy(np.asarray(inputs[k])) for k in INPUT_NAMES[:-1]}
    example["noise"] = torch.from_numpy(noise)
    diff = np.abs(full(**example).numpy() - reference_actions).max()
    print(f"traceable wrapper vs PyTorch sample_actions: max |diff| = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("the traceable wrapper diverges from the PyTorch model")

    t0 = time.perf_counter()
    ov_model = ov.convert_model(full, example_input=tuple(example.values()))
    for port, (name, value) in zip(ov_model.inputs, example.items()):
        port.get_tensor().set_names({name})
        port.get_node().set_layout(ov.Layout("N..."))
    ov_model.reshape({port: list(value.shape) for port, value in zip(ov_model.inputs, example.values())})
    ov_model.outputs[0].get_tensor().set_names({"actions"})
    print(f"converted in {time.perf_counter() - t0:.0f}s, saving {xml_path}")
    Path(xml_path).parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    del ov_model
    gc.collect()


def compress_mixed(model, int4_pattern=r"qwen_expert\.model\.layers\.\d+\.mlp\.experts/", group_size=128, keep_min_dim=64):
    """Weight-only compression: INT4 (asymmetric, grouped) for the action-expert routed MoE weights,
    INT8 (asymmetric, per channel) for the other MatMul weights.

    Weights with a dimension below keep_min_dim (MoE routers, state and action projections),
    convolutions and embeddings stay in FP16.
    """
    import re
    from collections import Counter

    import nncf
    from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

    matmuls = {}
    for op in model.get_ops():
        if op.get_type_name() != "MatMul":
            continue
        w = op.input_value(1).get_node()
        while w.get_type_name() == "Convert":
            w = w.input_value(0).get_node()
        shape = list(w.get_output_partial_shape(0).to_shape()) if w.get_type_name() == "Constant" else []
        if len(shape) == 2:
            matmuls[op.get_friendly_name()] = min(shape)
    keep = {name for name, dim in matmuls.items() if dim < keep_min_dim}
    int4 = {name for name in matmuls if name not in keep and re.search(int4_pattern, name)}
    print(f"{len(matmuls)} MatMuls with constant weights: {len(int4)} INT4, {len(matmuls) - len(int4) - len(keep)} INT8, {len(keep)} FP16")
    if not int4:
        raise ValueError(f"no MatMul matches {int4_pattern!r}")

    model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=group_size,
        ratio=1.0,
        all_layers=True,
        ignored_scope=nncf.IgnoredScope(names=sorted(set(matmuls) - int4), types=["Convolution", "Gather"], validate=False),
        advanced_parameters=AdvancedCompressionParameters(group_size_fallback_mode=nncf.GroupSizeFallbackMode.IGNORE),
    )
    # NNCF can give the decompression subgraphs of shared weights the same friendly name; the INT8 pass needs unique ones
    seen = Counter()
    for op in model.get_ops():
        seen[op.get_friendly_name()] += 1
        if seen[op.get_friendly_name()] > 1:
            op.set_friendly_name(f"{op.get_friendly_name()}_{seen[op.get_friendly_name()]}")
    return nncf.compress_weights(
        model, mode=nncf.CompressWeightsMode.INT8_ASYM, ignored_scope=nncf.IgnoredScope(names=sorted(keep), types=["Convolution", "Gather"], validate=False)
    )


class OVLingBotPolicy:
    """Runs the converted model: numpy inputs from LingBotPolicy.preprocess -> normalized actions."""

    def __init__(self, xml_path, device, core=None):
        import openvino as ov

        core = core or ov.Core()
        t0 = time.perf_counter()
        self.compiled = core.compile_model(xml_path, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.compile_time = time.perf_counter() - t0
        self.request = self.compiled.create_infer_request()
        self.noise_shape = tuple(self.compiled.input("noise").get_shape())

    def __call__(self, inputs, noise=None):
        if noise is None:
            noise = np.random.standard_normal(self.noise_shape).astype(np.float32)
        feed = {name: np.ascontiguousarray(inputs[name]) for name in INPUT_NAMES[:-1]}
        feed["noise"] = np.ascontiguousarray(noise, dtype=np.float32)
        return self.request.infer(feed)[0].copy()

    def benchmark(self, inputs, noise, warmup=3, iterations=20):
        for _ in range(warmup):
            self(inputs, noise)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self(inputs, noise)
            times.append(time.perf_counter() - t0)
        return 1000 * float(np.median(times))


def cos_sim(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
