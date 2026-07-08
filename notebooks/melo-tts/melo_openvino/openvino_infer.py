"""
MeloTTS OpenVINO 推理引擎。

加载 model_convert.py 生成的两个子模型 (melotts_enc.xml / melotts_dec.xml)，
在 Python 层完成 length regulator (VITS 的 generate_path) 与隐变量采样，
对外暴露与 PyTorch `SynthesizerTrn.infer` 兼容的接口。
"""

import os

import numpy as np
import openvino as ov

# 与 model_convert.py 保持一致的时长预测噪声尺度
DEFAULT_NOISE_SCALE_W = 0.8


def _generate_path(w_ceil, x_mask):
    """VITS length regulator 的 NumPy 实现。

    w_ceil: [1, 1, T_x] 每个音素的帧数 (向上取整)
    x_mask: [1, 1, T_x]
    返回 attn [T_y, T_x] 与 T_y。
    """
    duration = (w_ceil[0, 0] * x_mask[0, 0]).astype(np.int64)
    cum = np.cumsum(duration)
    T_y = int(cum[-1])
    idx = np.arange(T_y)[None, :]
    path = (idx < cum[:, None]).astype(np.float32)  # [T_x, T_y]
    path[1:] = path[1:] - path[:-1]
    return path.T, T_y  # [T_y, T_x]


class OVSynthesizer:
    """OpenVINO 版 SynthesizerTrn，接口与 torch 版 `infer` 对齐。"""

    def __init__(self, model_dir, device="CPU"):
        self.core = ov.Core()
        enc_path = os.path.join(model_dir, "melotts_enc.xml")
        dec_path = os.path.join(model_dir, "melotts_dec.xml")
        if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            raise FileNotFoundError(
                f"未找到 OpenVINO IR 模型，请先运行 model_convert.py。查找路径: {model_dir}"
            )
        self.enc = self.core.compile_model(enc_path, device)
        self.dec = self.core.compile_model(dec_path, device)
        self.device = device

    def infer(self, x, x_lengths, sid, tone, language, bert, ja_bert,
              noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8,
              max_len=None, sdp_ratio=0.0):
        """输入为 numpy 数组 (带 batch 维)，返回波形 numpy 数组 [1, 1, L]。"""
        T = x.shape[1]
        sdp_noise = (np.random.randn(1, 2, T).astype(np.float32) * noise_scale_w)

        enc_out = self.enc({
            "x": x.astype(np.int64),
            "x_lengths": x_lengths.astype(np.int64),
            "sid": sid.astype(np.int64),
            "tone": tone.astype(np.int64),
            "language": language.astype(np.int64),
            "bert": bert.astype(np.float32),
            "ja_bert": ja_bert.astype(np.float32),
            "sdp_noise": sdp_noise,
        })
        logw_sdp = enc_out[self.enc.output(0)]
        logw_dp = enc_out[self.enc.output(1)]
        m_p = enc_out[self.enc.output(2)]
        logs_p = enc_out[self.enc.output(3)]
        x_mask = enc_out[self.enc.output(4)]
        g = enc_out[self.enc.output(5)]

        logw = logw_sdp * sdp_ratio + logw_dp * (1 - sdp_ratio)
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)

        attn, T_y = _generate_path(w_ceil, x_mask)  # [T_y, T_x]
        m_p_exp = (attn @ m_p[0].T).T[None]         # [1, d, T_y]
        logs_p_exp = (attn @ logs_p[0].T).T[None]
        y_mask = np.ones((1, 1, T_y), dtype=np.float32)

        z_p = m_p_exp + np.random.randn(*m_p_exp.shape).astype(np.float32) \
            * np.exp(logs_p_exp) * noise_scale

        if max_len is not None:
            z_p = z_p[:, :, :max_len]
            y_mask = y_mask[:, :, :max_len]

        dec_out = self.dec({
            "z_p": z_p.astype(np.float32),
            "y_mask": y_mask.astype(np.float32),
            "g": g.astype(np.float32),
        })
        audio = dec_out[self.dec.output(0)]
        return audio
