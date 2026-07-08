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
    import os
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert
    from .chinese_mix import get_bert_feature as zh_mix_en_bert
    from .spanish_bert import get_bert_feature as sp_bert
    from .french_bert import get_bert_feature as fr_bert
    from .korean import get_bert_feature as kr_bert

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

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert, 
                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, "KR": kr_bert}

    # 解析实际加载路径：如果指定了 bert_dir 且本地存在，则用本地路径
    model_id = lang_bert_id_map.get(language)
    if bert_dir and model_id:
        local_path = os.path.join(bert_dir, model_id.replace("/", "--"))
        if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            model_id = local_path

    # 调用对应语言的 bert feature 函数
    func = lang_bert_func_map[language]
    if language == "ZH":
        bert = func(norm_text, word2ph, device, model_id=model_id)
    elif language in ("EN", "FR", "SP", "ES"):
        bert = func(norm_text, word2ph, device, model_id_override=model_id)
    elif language == "JP":
        bert = func(norm_text, word2ph, device, model_id=model_id)
    elif language == "ZH_MIX_EN":
        bert = func(norm_text, word2ph, device, model_id=model_id)
    elif language == "KR":
        bert = func(norm_text, word2ph, device, model_id=model_id)
    else:
        bert = func(norm_text, word2ph, device)
    return bert
