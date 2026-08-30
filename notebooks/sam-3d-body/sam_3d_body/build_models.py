# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")

    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialize the model.  At this point:
    #   • MHR TorchScript weights are loaded from mhr_path inside MHRHead.__init__()
    #   • hand_pose_comps_ori is created by cloning hand_pose_comps from MHR
    #   • DINOv3 mask_token is zero-initialised by the hub's init_weights()
    # All three sources are correct; they just aren't in the main checkpoint.
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # -----------------------------------------------------------------------
    # Inject parameters that the model correctly initialises from sources
    # OTHER than the main checkpoint, so that load_state_dict does not
    # report them as missing.  Three known categories:
    #
    #   1. *.mhr.*
    #      TorchScript MHR sub-model weights.  Loaded from `mhr_path`
    #      directly inside MHRHead.__init__() via torch.jit.load().
    #      They are intentionally absent from the main checkpoint.
    #
    #   2. *.hand_pose_comps_ori
    #      Derived by cloning hand_pose_comps after MHR loads
    #      (sam3d_body.py ~line 59/74).  This parameter was added to the
    #      architecture after the checkpoint was saved.
    #
    #   3. backbone.encoder.mask_token
    #      The DINOv3 hub added this parameter after the checkpoint was
    #      saved.  The hub zeroes it in init_weights() (pretrained=False
    #      path), so the value is already correct.
    # -----------------------------------------------------------------------
    _EXPECTED_MISSING_PATTERNS = (
        ".mhr.",
        ".hand_pose_comps_ori",
        "backbone.encoder.mask_token",
    )
    model_init_state = model.state_dict()
    injected_keys = []
    for key, val in model_init_state.items():
        if key not in state_dict and any(pat in key for pat in _EXPECTED_MISSING_PATTERNS):
            state_dict[key] = val.clone()
            injected_keys.append(key)

    if injected_keys:
        mhr_count = sum(1 for k in injected_keys if ".mhr." in k)
        non_mhr = [k for k in injected_keys if ".mhr." not in k]
        print(
            f"[load_sam_3d_body] Injected {len(injected_keys)} model-initialised keys "
            f"(not in main checkpoint by design):\n"
            f"  • {mhr_count} MHR TorchScript params  (source: {mhr_path or 'mhr_path'})\n"
            + "".join(f"  • {k}\n" for k in non_mhr)
        )
    else:
        mhr_count = 0

    load_state_dict(model, state_dict, strict=False)

    # -----------------------------------------------------------------------
    # Verification: confirm that every key coming from the checkpoint was
    # matched to a model parameter, and flag any genuinely unexpected
    # mismatches.
    # -----------------------------------------------------------------------
    checkpoint_keys = set(state_dict.keys()) - set(injected_keys)
    model_keys = set(model_init_state.keys())
    unmatched_ckpt_keys = checkpoint_keys - model_keys
    unmatched_model_keys = (model_keys - checkpoint_keys) - set(injected_keys)

    if unmatched_ckpt_keys:
        print(
            f"[load_sam_3d_body] WARNING: {len(unmatched_ckpt_keys)} checkpoint keys "
            f"have no corresponding model parameter (unexpected keys):\n  "
            + "\n  ".join(sorted(unmatched_ckpt_keys))
        )
    if unmatched_model_keys:
        print(
            f"[load_sam_3d_body] WARNING: {len(unmatched_model_keys)} model parameters "
            f"were not found in the checkpoint and are not in the expected-missing list:\n  "
            + "\n  ".join(sorted(unmatched_model_keys))
        )
    if not unmatched_ckpt_keys and not unmatched_model_keys:
        print(
            f"[load_sam_3d_body] ✓ All {len(checkpoint_keys)} checkpoint keys matched "
            f"model parameters.  {len(injected_keys)} keys supplied by model initialisation "
            f"({mhr_count} MHR + {len(injected_keys) - mhr_count} other)."
        )

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
