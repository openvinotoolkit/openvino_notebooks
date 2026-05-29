# OmniVoice Text-to-Speech with OpenVINO™

[OmniVoice](https://github.com/k2-fsa/OmniVoice) is a state-of-the-art zero-shot
text-to-speech model from the next-generation Kaldi team at Xiaomi AI Lab. It
supports **600+ languages**, voice cloning from a short reference clip, and
voice design via natural-language attribute prompts.

In this tutorial we convert OmniVoice — including its 0.6B Qwen3 backbone, the
HiggsAudio v2 audio tokenizer, and Whisper-large-v3-turbo (used for reference
audio auto-transcription) — to OpenVINO Intermediate Representation and run the
full pipeline on Intel CPU and GPU.

OmniVoice differs from typical TTS pipelines in that the LLM is invoked **once
per diffusion step** (default 32 steps) on the *full* sequence — there is no
KV-cache to reuse. The 32-step iterative masked-prediction loop, classifier-
free guidance, and Gumbel sampling all live in Python and call into the
compiled OpenVINO LLM each step. The OmniVoice helper provided in this notebook
keeps the public API identical to the upstream package:

```python
ov_model = OVOmniVoice.from_pretrained("ov_model", llm_device="GPU")
audios = ov_model.generate(
    text="Hello, this is OmniVoice on OpenVINO.",
    language="English",
    instruct="female, young adult",
)
```

The converted pipeline contains four sub-models:

| Model | Role |
|---|---|
| `openvino_llm_model.xml` | Qwen3-0.6B + audio embeddings + audio_heads (fused, INT8 optional) |
| `openvino_audio_encoder.xml` | HiggsAudio v2 encoder (waveform → 8-codebook tokens) |
| `openvino_audio_decoder.xml` | HiggsAudio v2 decoder (codes → waveform) |
| `whisper/` | Whisper-large-v3-turbo (only used when ref_text is empty) |

Notebook outline:

1. Install dependencies (`omnivoice`, `nncf`, `openvino`).
2. Convert the model to OpenVINO IR with an INT8-quantization toggle.
3. Choose CPU/GPU per sub-model.
4. Run Voice Design and Voice Clone inference; compare against the original
   PyTorch model.
5. Launch the interactive Gradio demo (Voice Clone + Voice Design tabs).

## Storage layout

The original PyTorch checkpoints (used only during conversion) and the
OpenVINO IR are kept in **separate** directories:

```
notebooks/omnivoice/
├── pt_models/                      # original PT weights — only needed at convert time
│   ├── OmniVoice/                  # k2-fsa/OmniVoice snapshot (~3 GB)
│   └── whisper-large-v3-turbo/     # OpenAI Whisper snapshot (~3 GB)
└── ov_model_int8/                  # self-contained OV runtime dir (~1.5 GB)
    ├── openvino_llm_model.{xml,bin}
    ├── openvino_audio_encoder.{xml,bin}
    ├── openvino_audio_decoder.{xml,bin}
    ├── whisper/openvino_{encoder,decoder}_model.{xml,bin}
    └── (config.json, tokenizer.json, audio_tokenizer/, …)
```

After conversion, `pt_models/` is safe to delete: the runtime
`OVOmniVoice.from_pretrained(<ov_dir>)` reads only from the OV folder and
allocates **zero PyTorch model weights**.

## Notes

- `transformers >= 5.3` is required (OmniVoice uses
  `HiggsAudioV2TokenizerModel`, added in transformers 5.x).
- The first run downloads ~3 GB of weights for OmniVoice and ~3 GB for Whisper.
- INT8 weight compression halves the LLM size and gives a meaningful speedup
  on CPU; audio fidelity is preserved because the convolutional codec stays in
  FP16.

## Installation

This is a self-contained example that relies solely on its own code. We
recommend running the notebook in a virtual environment. For details, please
refer to the
[Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
