from .symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device, bert_dir=None):
    """使用 OpenVINO BERT 提取文本特征。

    bert_dir 应指向包含 <model_id_safe>/bert_ov/bert.xml 的目录。
    """
    import os
    from ..bert_ov import get_bert_feature

    # 语言到默认 BERT model_id 的映射
    lang_bert_id_map = {
        "ZH": "hfl/chinese-roberta-wwm-ext-large",
        "EN": "bert-base-uncased",
        "JP": "tohoku-nlp/bert-base-japanese-v3",
        "ZH_MIX_EN": "bert-base-multilingual-uncased",
        "FR": "dbmdz/bert-base-french-europeana-cased",
        "SP": "dccuchile/bert-base-spanish-wwm-uncased",
        "ES": "dccuchile/bert-base-spanish-wwm-uncased",
        "KR": "kykim/bert-kor-base",
    }

    model_id = lang_bert_id_map.get(language)
    if not model_id:
        raise ValueError(f"不支持的语言: {language}")

    # 解析模型目录
    safe_name = model_id.replace("/", "--")
    if bert_dir:
        model_dir = os.path.join(bert_dir, safe_name)
    else:
        raise ValueError("必须提供 bert_dir 以定位 OpenVINO BERT 模型")

    return get_bert_feature(norm_text, word2ph, model_dir, device="CPU")
