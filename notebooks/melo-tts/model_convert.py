"""
MeloTTS -> OpenVINO IR conversion script

MeloTTS is a VITS-based TTS model. The compute flow of SynthesizerTrn.infer is:
    1. emb_g(sid)                          -> speaker embedding g
    2. enc_p(...)  (TextEncoder)           -> x, m_p, logs_p, x_mask
    3. sdp(reverse) / dp                   -> logw (duration prediction, with random noise)
    4. length regulator (generate_path)    -> attn, expands m_p / logs_p  (data-dependent dynamic length)
    5. z_p = m_p + randn * exp(logs_p)*ns  -> sampled latent
    6. flow(reverse)                       -> z
    7. dec (HiFiGAN Generator)             -> waveform

Following the skill's splitting principle, the "data-dependent, dynamic-length" length
regulator is kept in the Python layer (NumPy implementation), while the network body is
split into two statically traceable OpenVINO sub-models:

    melotts_enc.xml : emb_g + enc_p + sdp + dp   (text side, outputs duration and prior distribution)
    melotts_dec.xml : flow(reverse) + dec        (acoustic side, latent -> waveform)

All random noise is passed in explicitly as inputs, so the converted shapes vary
dynamically with sequence length and the results are reproducible.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import openvino as ov

from melo_torch.api import TTS


LANG_TO_MODEL_NAME = {
    "EN": "myshell-ai/MeloTTS-English",
    "EN_V2": "myshell-ai/MeloTTS-English-v2",
    "EN_NEWEST": "myshell-ai/MeloTTS-English-v3",
    "FR": "myshell-ai/MeloTTS-French",
    "JP": "myshell-ai/MeloTTS-Japanese",
    "ES": "myshell-ai/MeloTTS-Spanish",
    "ZH": "myshell-ai/MeloTTS-Chinese",
    "KR": "myshell-ai/MeloTTS-Korean",
}

# Duration-prediction noise scale fixed at conversion time (the same default is used at inference)
DEFAULT_NOISE_SCALE_W = 0.8


class EncWrapper(nn.Module):
    """Text encoding + duration prediction sub-model.

    Outputs logw_sdp / logw_dp (sdp_ratio is not fused internally, keeping runtime flexibility),
    plus the prior distribution parameters m_p, logs_p, the text mask x_mask, and speaker embedding g.
    The random noise sdp_noise is passed in explicitly as input (already multiplied by noise_scale_w externally).
    """

    def __init__(self, model):
        super().__init__()
        self.emb_g = model.emb_g
        self.enc_p = model.enc_p
        self.sdp = model.sdp
        self.dp = model.dp

    def _sdp_reverse(self, x, x_mask, g, z):
        sdp = self.sdp
        x = torch.detach(x)
        x = sdp.pre(x)
        g = torch.detach(g)
        x = x + sdp.cond(g)
        x = sdp.convs(x, x_mask)
        x = sdp.proj(x) * x_mask

        flows = list(reversed(sdp.flows))
        flows = flows[:-2] + [flows[-1]]  # remove one useless vflow
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        z0, _z1 = torch.split(z, [1, 1], 1)
        return z0

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert, sdp_noise):
        g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        x_enc, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        logw_sdp = self._sdp_reverse(x_enc, x_mask, g, sdp_noise)
        logw_dp = self.dp(x_enc, x_mask, g=g)
        return logw_sdp, logw_dp, m_p, logs_p, x_mask, g


class DecWrapper(nn.Module):
    """flow(reverse) + HiFiGAN decoding sub-model: latent z_p -> waveform."""

    def __init__(self, model):
        super().__init__()
        self.flow = model.flow
        self.dec = model.dec

    def forward(self, z_p, y_mask, g):
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z * y_mask, g=g)
        return o


class BertHiddenStateWrapper(nn.Module):
    """Extract hidden_states[-3] from an AutoModelForMaskedLM.

    MeloTTS only uses the hidden state of layer -3 as text features.
    This wrapper wraps the BERT model into a single-output module to ease OpenVINO conversion.
    """

    def __init__(self, bert_model):
        super().__init__()
        # Only the BERT backbone (bert/roberta) is needed, not the MLM head
        if hasattr(bert_model, 'bert'):
            self.bert = bert_model.bert
        elif hasattr(bert_model, 'roberta'):
            self.bert = bert_model.roberta
        else:
            self.bert = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        # Return the hidden state of the 3rd-from-last layer
        return outputs.hidden_states[-3]


def build_example_inputs(tts, text, speaker_id):
    """Run the text frontend to build a set of real-shaped example inputs for tracing."""
    from melo_torch import utils

    language = tts.language
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
        text, language, tts.hps, "cpu", tts.symbol_to_id, bert_dir=tts.bert_dir
    )
    x = phones.unsqueeze(0)
    tones = tones.unsqueeze(0)
    lang_ids = lang_ids.unsqueeze(0)
    bert = bert.unsqueeze(0)
    ja_bert = ja_bert.unsqueeze(0)
    x_lengths = torch.LongTensor([phones.size(0)])
    sid = torch.LongTensor([speaker_id])

    T = phones.size(0)
    sdp_noise = torch.randn(1, 2, T) * DEFAULT_NOISE_SCALE_W
    enc_inputs = (x, x_lengths, sid, tones, lang_ids, bert, ja_bert, sdp_noise)
    return enc_inputs


# Language -> list of BERT model_ids that need conversion
LANG_BERT_IDS = {
    "ZH": ["hfl/chinese-roberta-wwm-ext-large", "bert-base-multilingual-uncased"],
    "EN": ["bert-base-uncased"],
    "EN_V2": ["bert-base-uncased"],
    "EN_NEWEST": ["bert-base-uncased"],
    "JP": ["tohoku-nlp/bert-base-japanese-v3"],
    "FR": ["dbmdz/bert-base-french-europeana-cased"],
    "ES": ["dccuchile/bert-base-spanish-wwm-uncased"],
    "KR": ["kykim/bert-kor-base"],
}


# The tokenizer only needs these files to load
_TOKENIZER_FILES = [
    "tokenizer_config.json", "tokenizer.json",
    "vocab.txt", "special_tokens_map.json",
]


def _convert_bert_models(language, bert_dir, output_dir):
    """Convert the BERT models required by each language to OpenVINO IR.

    Output directory layout (inference-required files only):
        output_dir/bert/<model_id_safe>/
            model.xml, model.bin    — OV BERT IR
            tokenizer_config.json   — tokenizer config
            tokenizer.json          — fast tokenizer
            vocab.txt               — vocabulary
            special_tokens_map.json — special token mapping
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    bert_ids = LANG_BERT_IDS.get(language, [])
    if not bert_ids:
        return

    dst_bert_root = os.path.join(output_dir, "bert")
    os.makedirs(dst_bert_root, exist_ok=True)

    for model_id in bert_ids:
        safe_name = model_id.replace("/", "--")
        dst_model_dir = os.path.join(dst_bert_root, safe_name)
        dst_xml = os.path.join(dst_model_dir, "model.xml")

        if os.path.isfile(dst_xml):
            print(f"      BERT {safe_name}: 已存在，跳过")
            continue

        # Determine the load path: prefer local bert_dir
        load_path = model_id
        if bert_dir:
            local = os.path.join(bert_dir, safe_name)
            if os.path.isdir(local):
                load_path = local

        print(f"      BERT {safe_name}: 加载并转换...")

        # Load PyTorch BERT
        pt_model = AutoModelForMaskedLM.from_pretrained(load_path)
        pt_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Wrap into a module that only outputs hidden_states[-3]
        wrapper = BertHiddenStateWrapper(pt_model).eval()

        # Build example inputs
        sample = tokenizer("Hello world", return_tensors="pt")
        example_input = (
            sample["input_ids"],
            sample["attention_mask"],
            sample.get("token_type_ids", torch.zeros_like(sample["input_ids"])),
        )

        # 转换
        with torch.no_grad():
            ov_bert = ov.convert_model(wrapper, example_input=example_input)

        # Set dynamic shapes
        bert_input_names = ["input_ids", "attention_mask", "token_type_ids"]
        bert_dyn = {n: ov.PartialShape([1, -1]) for n in bert_input_names}
        for port, name in zip(ov_bert.inputs, bert_input_names):
            port.get_node().set_friendly_name(name)
            port.get_tensor().set_names({name})
        ov_bert.reshape(bert_dyn)

        # Save OV IR
        os.makedirs(dst_model_dir, exist_ok=True)
        ov.save_model(ov_bert, dst_xml)

        # Save only the required tokenizer files (excluding pytorch_model.bin etc.)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer.save_pretrained(tmp)
            for fname in _TOKENIZER_FILES:
                src = os.path.join(tmp, fname)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy2(src, os.path.join(dst_model_dir, fname))

        print(f"      BERT {safe_name}: 已保存 IR + tokenizer")

        # Free memory
        del pt_model, wrapper, ov_bert


def _remove_weight_norm(module):
    """Fold weight_norm (weight_g / weight_v) into a single weight tensor across a module tree.

    The flow (``WN``) and HiFiGAN generator (``dec``) wrap every convolution with ``weight_norm``.
    Leaving it in place forces the tracer and OpenVINO constant-folding to recompute each weight
    norm, which makes decoder conversion take many minutes. Folding it is numerically equivalent
    and dramatically speeds up conversion. A parent module's ``remove_weight_norm`` already strips
    its children, so a redundant call on a child raises and is safely ignored.
    """
    for m in module.modules():
        remove = getattr(m, "remove_weight_norm", None)
        if callable(remove):
            try:
                remove()
            except (ValueError, RuntimeError):
                pass


def convert(language, output_dir, device_check=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = LANG_TO_MODEL_NAME[language]
    model_dir = os.path.join(base_dir, "models", model_name.split("/")[-1])
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "checkpoint.pth")
    bert_dir = os.path.join(model_dir, "bert")
    bert_dir = bert_dir if os.path.isdir(bert_dir) else None

    if output_dir is None:
        output_dir = os.path.join(model_dir, "openvino")
    os.makedirs(output_dir, exist_ok=True)

    enc_xml = os.path.join(output_dir, "melotts_enc.xml")
    dec_xml = os.path.join(output_dir, "melotts_dec.xml")

    # Skip the expensive PyTorch load + trace + conversion if both IR sub-models already exist.
    # Auxiliary artifacts (config.json + BERT IR) have their own existence checks, so we still
    # make sure they are present before returning.
    if os.path.isfile(enc_xml) and os.path.isfile(dec_xml):
        import shutil
        print(f"OpenVINO IR 已存在，跳过转换:\n      {enc_xml}\n      {dec_xml}")
        dst_config = os.path.join(output_dir, "config.json")
        if not os.path.isfile(dst_config) and os.path.isfile(config_path):
            shutil.copy2(config_path, dst_config)
            print(f"      已复制: config.json")
        _convert_bert_models(language, bert_dir, output_dir)
        print("转换完成（复用已有 IR）。")
        return

    print(f"[1/5] 加载 PyTorch 模型: {model_name}")
    tts = TTS(language=language, device="cpu", config_path=config_path,
              ckpt_path=ckpt_path, bert_dir=bert_dir)
    tts.model.eval()

    speaker_id = list(tts.hps.data.spk2id.values())[0]
    enc = EncWrapper(tts.model).eval()
    dec = DecWrapper(tts.model).eval()

    print("[2/5] 构造样例输入 (运行文本前端)")
    sample_text = "你好，欢迎使用 MeloTTS 的 OpenVINO 加速版本。"
    enc_inputs = build_example_inputs(tts, sample_text, speaker_id)

    print("[3/5] 转换 encoder 子模型 (enc_p + sdp + dp)")
    with torch.no_grad():
        enc_outputs = enc(*enc_inputs)
        ov_enc = ov.convert_model(enc, example_input=enc_inputs)

    enc_input_names = ["x", "x_lengths", "sid", "tone", "language",
                       "bert", "ja_bert", "sdp_noise"]
    enc_dyn = {
        "x": ov.PartialShape([1, -1]),
        "x_lengths": ov.PartialShape([1]),
        "sid": ov.PartialShape([1]),
        "tone": ov.PartialShape([1, -1]),
        "language": ov.PartialShape([1, -1]),
        "bert": ov.PartialShape([1, 1024, -1]),
        "ja_bert": ov.PartialShape([1, 768, -1]),
        "sdp_noise": ov.PartialShape([1, 2, -1]),
    }
    for port, name in zip(ov_enc.inputs, enc_input_names):
        port.get_node().set_friendly_name(name)
        port.get_tensor().set_names({name})
    ov_enc.reshape({name: enc_dyn[name] for name in enc_input_names})
    enc_xml = os.path.join(output_dir, "melotts_enc.xml")
    ov.save_model(ov_enc, enc_xml)
    print(f"      已保存: {enc_xml}")

    # decoder trace uses short fixed-length dummy inputs to avoid the trace hanging on HiFiGAN's heavy upsampling.
    # Once the IR is set to dynamic shapes, inference supports any T_y; the length here only affects trace speed, not correctness.
    print("[4/5] 转换 decoder 子模型 (flow + dec)")
    # Fold weight_norm in flow + HiFiGAN before tracing; otherwise convert_model spends many
    # minutes constant-folding the weight_norm subgraphs. This is numerically equivalent.
    _remove_weight_norm(dec)
    logw_sdp, logw_dp, m_p, logs_p, x_mask, g = [t.numpy() for t in enc_outputs]
    inter_channels = m_p.shape[1]
    gin_channels = g.shape[1]

    T_y_trace = 32  # short sequence for tracing, to avoid HiFiGAN upsampling hangs
    z_p_trace = np.random.randn(1, inter_channels, T_y_trace).astype(np.float32)
    y_mask_trace = np.ones((1, 1, T_y_trace), dtype=np.float32)

    # Also keep a real-length copy of the data for later numerical validation
    sdp_ratio = 0.2
    length_scale = 1.0
    noise_scale = 0.667
    logw = logw_sdp * sdp_ratio + logw_dp * (1 - sdp_ratio)
    w = np.exp(logw) * x_mask * length_scale
    w_ceil = np.ceil(w)
    duration = w_ceil[0, 0].astype(np.int64)
    cum = np.cumsum(duration)
    T_y = int(cum[-1])
    idx = np.arange(T_y)[None, :]
    path = (idx < cum[:, None]).astype(np.float32)
    path[1:] = path[1:] - path[:-1]
    attn = path.T
    m_p_exp = (attn @ m_p[0].T).T[None]
    logs_p_exp = (attn @ logs_p[0].T).T[None]
    y_mask_full = np.ones((1, 1, T_y), dtype=np.float32)
    z_p_full = m_p_exp + np.random.randn(*m_p_exp.shape).astype(np.float32) * np.exp(logs_p_exp) * noise_scale

    dec_inputs = (
        torch.from_numpy(z_p_trace),
        torch.from_numpy(y_mask_trace),
        torch.from_numpy(g.astype(np.float32)),
    )
    with torch.no_grad():
        ov_dec = ov.convert_model(dec, example_input=dec_inputs)

    dec_input_names = ["z_p", "y_mask", "g"]
    dec_dyn = {
        "z_p": ov.PartialShape([1, inter_channels, -1]),
        "y_mask": ov.PartialShape([1, 1, -1]),
        "g": ov.PartialShape([1, gin_channels, 1]),
    }
    for port, name in zip(ov_dec.inputs, dec_input_names):
        port.get_node().set_friendly_name(name)
        port.get_tensor().set_names({name})
    ov_dec.reshape({name: dec_dyn[name] for name in dec_input_names})
    dec_xml = os.path.join(output_dir, "melotts_dec.xml")
    ov.save_model(ov_dec, dec_xml)
    print(f"      已保存: {dec_xml}")

    # Copy the auxiliary files needed for inference to the output directory (config.json)
    import shutil
    dst_config = os.path.join(output_dir, "config.json")
    if not os.path.isfile(dst_config):
        shutil.copy2(config_path, dst_config)
        print(f"      已复制: config.json")

    # Convert BERT models to OpenVINO IR and copy the tokenizer
    _convert_bert_models(language, bert_dir, output_dir)

    print("[5/5] 校验 (CPU 编译并对比数值)")
    core = ov.Core()
    c_enc = core.compile_model(ov_enc, "CPU")
    c_dec = core.compile_model(ov_dec, "CPU")
    enc_ref = {n: i.numpy() for n, i in zip(enc_input_names, enc_inputs)}
    ov_enc_out = c_enc(enc_ref)
    # Compare m_p (index 2) against torch
    m_p_ov = list(ov_enc_out.values())[2]
    diff = np.abs(m_p_ov - m_p).max()
    print(f"      encoder m_p 最大误差: {diff:.3e}")

    ov_dec_out = c_dec({"z_p": z_p_full, "y_mask": y_mask_full, "g": g})
    audio = list(ov_dec_out.values())[0]
    print(f"      decoder 输出波形 shape: {audio.shape}")
    print("转换完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeloTTS -> OpenVINO IR 转换")
    parser.add_argument("--lang", "-l", type=str, default="ZH",
                        choices=list(LANG_TO_MODEL_NAME.keys()),
                        help="语言/模型 (默认 ZH)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="IR 输出目录 (默认 models/<MODEL>/openvino)")
    args = parser.parse_args()
    convert(args.lang, args.output_dir)
