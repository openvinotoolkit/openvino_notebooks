"""Small inference, data and metric utilities for the Polyglot-Lion tutorial.

The encoder wrapper follows Qwen3-ASR's Apache-2.0 implementation:
Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
See licensing/third-party-programs.txt in the repository root.
"""

import hashlib
import json
import re
from pathlib import Path
import statistics
import time
import unicodedata
from types import MethodType

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import openvino as ov
from datasets import Audio, load_dataset
from qwen_asr.inference.utils import parse_asr_output

MODEL_ID = "knoveleng/polyglot-lion-1.7b"
MODEL_REVISION = "9ef388b498182f50a5a7909e6e7a4e2d89a840c4"
DATASET_ID = "google/fleurs"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LANGUAGES = {"en_us": "English", "cmn_hans_cn": "Mandarin", "ta_in": "Tamil", "ms_my": "Malay"}
IR_FILES = (
    "config.json",
    "generation_config.json",
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
)

AUDIO_ENCODER_FIX = "eager-sdpa-full-utterance-valid-cnn-v1"


def restore_decoder_projections(compressed: ov.Model, reference: ov.Model) -> list[str]:
    """Keep attention and first-eight-layer decoder projection weights in FP16."""
    pattern = r"layers\.[0-7]\.|layers\.\d+\.self_attn\.[qkvo]_proj/"
    originals = {node.get_friendly_name(): node for node in reference.get_ordered_ops() if node.get_type_name() == "MatMul"}
    restored = []
    for node in compressed.get_ordered_ops():
        name = node.get_friendly_name()
        if node.get_type_name() == "MatMul" and re.search(pattern, name):
            if name not in originals:
                raise ValueError(f"Missing original decoder projection: {name}")
            node.input(1).replace_source_output(originals[name].input_value(1))
            restored.append(name)
    if not restored:
        raise ValueError("No matching decoder projections; check the pinned export")
    compressed.validate_nodes_and_infer_types()
    return restored


class FullContextAudioEncoder(torch.nn.Module):
    """Export full-utterance eager/SDPA attention after removing CNN padding."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features, valid_mask):
        encoder = self.encoder
        x = F.gelu(encoder.conv2d1(input_features.unsqueeze(1)))
        x = F.gelu(encoder.conv2d2(x))
        x = F.gelu(encoder.conv2d3(x))
        batch, channels, frequencies, frames = x.shape
        x = encoder.conv_out(x.permute(0, 3, 1, 2).reshape(batch, frames, channels * frequencies))
        x = x + encoder.positional_embedding.positional_embedding[:frames].unsqueeze(0).to(x.dtype)
        hidden = x[valid_mask]
        boundaries = torch.cat((torch.zeros(1, dtype=torch.int64, device=x.device), torch._shape_as_tensor(hidden)[:1]))
        for layer in encoder.layers:
            hidden = layer(hidden, cu_seqlens=boundaries)[0]
        hidden = encoder.ln_post(hidden)
        return encoder.proj2(encoder.act(encoder.proj1(hidden))).unsqueeze(0)


def export_audio_encoder(encoder, path: Path):
    """Convert the original eager/SDPA Qwen encoder with full-utterance attention."""
    if encoder.n_window != 50:
        raise ValueError("This export requires the pinned model's 100-frame CNN chunks")
    original_attention = encoder.config._attn_implementation
    if original_attention not in ("eager", "sdpa"):
        raise ValueError("This export matches eager/SDPA, not Flash Attention window semantics")
    encoder.config._attn_implementation = "eager"
    try:
        wrapper = FullContextAudioEncoder(encoder).eval()
        mask = torch.ones((8, 13), dtype=torch.bool)
        mask[-1, -3:] = False
        model = ov.convert_model(
            wrapper,
            example_input=(torch.zeros(8, 128, 100), mask),
            input=[("input_features", [-1, 128, -1]), ("valid_mask", [-1, -1])],
        )
        model.output(0).get_tensor().set_names({"last_hidden_state"})
        ov.save_model(model, path, compress_to_fp16=True)
    finally:
        encoder.config._attn_implementation = original_attention


def audio_windows(input_features, feature_mask, chunk_length: int, window_length: int):
    """Match original per-utterance CNN padding and attention-window boundaries."""
    if input_features.ndim != 2 or chunk_length < 1 or window_length % chunk_length:
        raise ValueError("Expected one mel spectrogram and integral chunks per attention window")
    length = input_features.shape[-1]
    if feature_mask is not None:
        mask = feature_mask.reshape(-1)
        if mask.numel() != length or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Expected one binary feature mask value per mel frame")
        length = int(mask.sum().item())
        if not torch.all(mask[:length] == 1) or not torch.all(mask[length:] == 0):
            raise ValueError("Audio feature mask must be valid frames followed by padding")
    if length < 1:
        raise ValueError("Cannot encode empty audio features")
    width = min(chunk_length, length)
    chunks = [chunk.T for chunk in input_features[:, :length].split(width, dim=-1)]
    padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
    lengths = torch.tensor([chunk.shape[0] for chunk in chunks])
    for _ in range(3):
        lengths = (lengths + 1) // 2
    frames = (width + 7) // 8
    valid = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    per_window = window_length // chunk_length
    for start in range(0, len(chunks), per_window):
        yield padded[start : start + per_window], valid[start : start + per_window]


def configure_audio_encoder(model):
    """Install the runtime adapter only for explicitly corrected encoder artifacts."""
    names = {port.get_any_name() for port in model.encoder.model.inputs}
    if names != {"input_features", "valid_mask"}:
        raise ValueError("Expected a corrected audio encoder with input_features and valid_mask")
    if model.config.model_type != "qwen3_asr":
        raise ValueError("The corrected encoder adapter supports only Qwen3-ASR")
    if model.encoder.request is not None:
        raise ValueError("Load with compile=False before configuring the corrected encoder")

    def chunked_forward(part, input_features, feature_mask, n_window, n_window_infer):
        outputs = []
        # The pinned reference's eager/SDPA attention ignores cu_seqlens; it is global.
        chunk_length = n_window * 2
        utterance_window = ((input_features.shape[-1] + chunk_length - 1) // chunk_length) * chunk_length
        for chunks, valid in audio_windows(input_features, feature_mask, chunk_length, utterance_window):
            result = part.request({"input_features": chunks, "valid_mask": valid})
            outputs.append(torch.from_numpy(result["last_hidden_state"].copy()).to(part.device))
        return torch.cat(outputs, dim=1)

    model.encoder.chunked_forward = MethodType(chunked_forward, model.encoder)


def sha256(path: Path) -> str:
    """Hash a file without loading model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(folder: Path) -> dict[str, str]:
    """Require a complete encoder/stateful-decoder IR and fingerprint its files."""
    return {name: sha256(folder / name) for name in IR_FILES}


def load_samples(configs: tuple[str, ...], count: int, directory: Path, start: int = 0) -> list[dict]:
    """Download a deterministic FLEURS slice or verify its exact cached selection.

    :param configs: Supported FLEURS language configurations.
    :param count: Number of unique recordings per language.
    :param directory: Local directory for WAV files and their manifest.
    :param start: Zero-based first dataset row to consider.
    :returns: Recordings with labels, provenance and SHA-256 hashes.
    """
    if not configs or len(set(configs)) != len(configs) or set(configs) - LANGUAGES.keys():
        raise ValueError("Select distinct supported FLEURS language configurations")
    if count < 1 or start < 0:
        raise ValueError("Sample count must be positive and start nonnegative")
    selection = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": "test",
        "configs": list(configs),
        "count": count,
        "start": start,
    }
    manifest = directory / "manifest.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached["selection"] != selection:
            raise ValueError("Cached sample selection differs; choose a different data directory")
        samples = cached["samples"]
    else:
        directory.mkdir(parents=True, exist_ok=True)
        samples = []
        seen_ids, seen_hashes = set(), set()
        for config in configs:
            dataset = load_dataset(
                DATASET_ID,
                config,
                split="test",
                streaming=True,
                revision=DATASET_REVISION,
            ).cast_column("audio", Audio(decode=False))
            selected = 0
            for index, row in enumerate(dataset):
                if index < start or (config, row["id"]) in seen_ids:
                    continue
                content = row["audio"].get("bytes")
                if not content:
                    raise ValueError(f"No embedded WAV data for {config}/{row['id']}")
                digest = hashlib.sha256(content).hexdigest()
                if digest in seen_hashes:
                    continue
                filename = f"{config}-{index}-{row['id']}.wav"
                (directory / filename).write_bytes(content)
                samples.append(
                    {
                        "config": config,
                        "language": LANGUAGES[config],
                        "id": row["id"],
                        "reference": row["transcription"],
                        "audio": filename,
                        "audio_sha256": digest,
                        "license": "CC-BY-4.0",
                    }
                )
                seen_ids.add((config, row["id"]))
                seen_hashes.add(digest)
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f"Insufficient eligible FLEURS data for {config}")
        manifest.write_text(json.dumps({"selection": selection, "samples": samples}, ensure_ascii=False, indent=2))
    if (
        len(samples) != len(configs) * count
        or len({(s["config"], s["id"]) for s in samples}) != len(samples)
        or len({s["audio_sha256"] for s in samples}) != len(samples)
        or any(sum(s["config"] == config for s in samples) != count for config in configs)
    ):
        raise ValueError("Incorrect cached sample count or duplicate sample identities")
    for sample in samples:
        if sha256(directory / sample["audio"]) != sample["audio_sha256"]:
            raise ValueError(f"Cached audio changed: {sample['audio']}")
    return samples


def prepare_inputs(processor, audio_path: Path) -> tuple[dict, float]:
    """Load/resample audio and build the original empty-system ASR prompt."""
    audio, rate = librosa.load(audio_path, sr=16000, mono=True)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty finite mono audio after resampling to 16000 Hz")
    prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt, audio=audio, sampling_rate=rate, return_tensors="pt")
    return inputs, len(audio) / rate


def generate(model, inputs: dict, backend: str, max_new_tokens: int = 512, **kwargs):
    """Generate greedily using identical masks/features for the two backends."""
    if backend not in ("pytorch", "openvino") or max_new_tokens < 1:
        raise ValueError("Select pytorch/openvino and a positive generation budget")
    with torch.inference_mode():
        if backend == "pytorch":
            return model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **kwargs)
        return model.generate(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            input_features_mask=inputs["feature_attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


def transcribe(model, processor, inputs: dict, backend: str, max_new_tokens: int = 512) -> dict:
    """Decode only newly generated tokens and record EOS completion."""
    start = time.perf_counter()
    output = generate(model, inputs=inputs, backend=backend, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - start
    sequences = output.sequences if hasattr(output, "sequences") else output
    ids = sequences[:, inputs["input_ids"].shape[1] :].cpu().tolist()[0]
    raw = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    language, text = parse_asr_output(raw)
    return {
        "transcription": text,
        "language": language,
        "token_ids": ids,
        "complete": bool(ids) and ids[-1] in (151643, 151645),
        "seconds": elapsed,
    }


def corpus_error(references: list[str], hypotheses: list[str], characters: bool = False) -> float:
    """Compute reference-weighted WER/CER, preserving Unicode combining marks."""
    if not references or len(references) != len(hypotheses):
        raise ValueError("Expected nonempty matching reference/hypothesis lists")

    def units(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = "".join(c if unicodedata.category(c)[0] in "LNM" or c.isspace() else " " for c in normalized)
        return list("".join(normalized.split())) if characters else normalized.split()

    edits, total = 0, 0
    for reference, hypothesis in zip(references, hypotheses):
        ref, hyp = units(reference), units(hypothesis)
        if not ref:
            raise ValueError("Reference is empty after normalization")
        previous = list(range(len(hyp) + 1))
        for index, word in enumerate(ref, start=1):
            current = [index]
            for other, actual in enumerate(hyp, start=1):
                current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (word != actual)))
            previous = current
        edits += previous[-1]
        total += len(ref)
    return edits / total


def logit_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Report finite, shape-checked differences without cosine-only approval."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError("Expected matching nonempty logits")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Logits must be finite")
    norms = np.linalg.norm(expected), np.linalg.norm(actual)
    cosine = (
        float(norms[0] == norms[1])
        if 0 in norms
        else float(
            np.dot(
                expected.ravel() / norms[0],
                actual.ravel() / norms[1],
            )
        )
    )
    delta = actual - expected
    return {
        "cosine": cosine,
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_absolute_error": float(np.abs(delta).max()),
        "top1_match": int(expected.argmax()) == int(actual.argmax()),
    }


def benchmark(infer, iterations: int = 3, warmup: int = 1) -> dict:
    """Time complete greedy generation, excluding model load/compile and audio I/O."""
    if iterations < 1 or warmup < 0:
        raise ValueError("Expected positive iterations and nonnegative warmup")
    for _ in range(warmup):
        if not infer()["complete"]:
            raise ValueError("Warmup generation did not reach EOS")
    outputs = [infer() for _ in range(iterations)]
    if any(not output["complete"] for output in outputs):
        raise ValueError("Benchmark generation did not reach EOS")
    return {
        "median_seconds": statistics.median(output["seconds"] for output in outputs),
        "tokens": len(outputs[-1]["token_ids"]),
        "repeat_tokens_identical": all(output["token_ids"] == outputs[0]["token_ids"] for output in outputs),
    }
