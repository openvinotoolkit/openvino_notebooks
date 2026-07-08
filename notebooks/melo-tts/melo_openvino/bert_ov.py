"""
OpenVINO BERT 特征提取器。

替代 PyTorch 版的 AutoModelForMaskedLM，提取 hidden_states[-3][0]
并按 word2ph 扩展到 phone 级别。Tokenizer 仍使用 transformers（纯 Python，无需 torch）。
"""

import os

import numpy as np
import openvino as ov

# 全局缓存，避免重复加载
_compiled_models = {}
_tokenizers = {}


def _get_tokenizer(model_dir):
    """加载 tokenizer（纯 Python，不依赖 torch）。"""
    if model_dir not in _tokenizers:
        from transformers import AutoTokenizer
        _tokenizers[model_dir] = AutoTokenizer.from_pretrained(model_dir)
    return _tokenizers[model_dir]


def _get_compiled_model(model_dir, device="CPU"):
    """加载编译后的 OpenVINO BERT 模型。"""
    key = (model_dir, device)
    if key not in _compiled_models:
        xml_path = os.path.join(model_dir, "model.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(
                f"未找到 BERT OpenVINO 模型: {xml_path}\n"
                f"请重新运行 model_convert.py 以转换 BERT 模型。"
            )
        core = ov.Core()
        _compiled_models[key] = core.compile_model(xml_path, device)
    return _compiled_models[key]


def get_bert_feature(text, word2ph, model_dir, device="CPU"):
    """提取 BERT 特征并扩展到 phone 级别。

    Args:
        text: 输入文本
        word2ph: 每个 word/token 对应的 phone 数量列表
        model_dir: BERT 模型目录 (含 bert_ov/bert.xml 和 tokenizer 文件)
        device: OpenVINO 设备

    Returns:
        phone_level_feature: numpy array [hidden_dim, total_phones]
    """
    tokenizer = _get_tokenizer(model_dir)
    compiled_model = _get_compiled_model(model_dir, device)

    inputs = tokenizer(text, return_tensors="np")

    result = compiled_model({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs.get("token_type_ids",
                                      np.zeros_like(inputs["input_ids"])),
    })
    # 输出是 hidden_states[-3]，shape [1, seq_len, hidden_dim]
    hidden = result[compiled_model.output(0)][0]  # [seq_len, hidden_dim]

    # 按 word2ph 扩展到 phone 级别
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = np.tile(hidden[i], (word2ph[i], 1))  # [n_phones, hidden_dim]
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)  # [total_phones, hidden_dim]
    return phone_level_feature.T  # [hidden_dim, total_phones]
