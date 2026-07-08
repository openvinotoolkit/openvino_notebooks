"""
Centralized download script for the models and data required to run MeloTTS.

Usage:
    python download.py                # Download everything needed for Chinese (ZH) (default)
    python download.py --lang ZH EN   # Download dependencies for Chinese and English
    python download.py --all          # Download models and data for all languages
"""

import argparse
import subprocess
import sys
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ============== Config ==============

# MeloTTS models (HuggingFace)
LANG_TO_HF_REPO = {
    "EN": "myshell-ai/MeloTTS-English",
    "EN_V2": "myshell-ai/MeloTTS-English-v2",
    "EN_NEWEST": "myshell-ai/MeloTTS-English-v3",
    "FR": "myshell-ai/MeloTTS-French",
    "JP": "myshell-ai/MeloTTS-Japanese",
    "ES": "myshell-ai/MeloTTS-Spanish",
    "ZH": "myshell-ai/MeloTTS-Chinese",
    "KR": "myshell-ai/MeloTTS-Korean",
}

# BERT models (per language)
LANG_TO_BERT = {
    "EN": ["bert-base-uncased"],
    "EN_V2": ["bert-base-uncased"],
    "EN_NEWEST": ["bert-base-uncased"],
    "FR": ["dbmdz/bert-base-french-europeana-cased"],
    "JP": ["tohoku-nlp/bert-base-japanese-v3"],
    "ES": ["dccuchile/bert-base-spanish-wwm-uncased"],
    "ZH": ["bert-base-multilingual-uncased", "hfl/chinese-roberta-wwm-ext-large"],
    "KR": ["kykim/bert-kor-base"],
}


def download_unidic():
    """Download MeCab's unidic dictionary (required for Japanese; loaded at MeloTTS import time)."""
    print("\n" + "=" * 60)
    print("[1/3] 检查 unidic 字典...")
    print("=" * 60)
    try:
        import unidic
        dicdir = unidic.DICDIR
        mecabrc = os.path.join(dicdir, "mecabrc")
        if os.path.exists(mecabrc):
            print(f"  ✓ unidic 字典已存在: {dicdir}")
            return
    except ImportError:
        print("  ! unidic 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unidic", "-q"])

    print("  下载 unidic 字典 (~526MB)...")
    # unidic download sometimes exits with code=1 due to a zip path issue but has actually extracted successfully
    result = subprocess.run([sys.executable, "-m", "unidic", "download"])
    # Verify success
    import importlib
    import unidic
    importlib.reload(unidic)
    mecabrc = os.path.join(unidic.DICDIR, "mecabrc")
    if os.path.exists(mecabrc):
        print(f"  ✓ unidic 字典下载完成: {unidic.DICDIR}")
    else:
        print(f"  ✗ unidic 字典可能未完整安装，请手动检查: {unidic.DICDIR}")

    # nltk data (needed for English g2p)
    print("  下载 nltk 数据...")
    import nltk
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print("  ✓ nltk averaged_perceptron_tagger_eng")


def download_models(languages):
    """Download MeloTTS checkpoint/config and the corresponding BERT models to ./models/{ModelName}/."""
    print("\n" + "=" * 60)
    print("[2/2] 下载模型...")
    print("=" * 60)
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    downloaded_bert = set()
    for lang in languages:
        if lang not in LANG_TO_HF_REPO:
            print(f"  ! 跳过未知语言: {lang}")
            continue

        repo_id = LANG_TO_HF_REPO[lang]
        model_name = repo_id.split("/")[-1]  # e.g. "MeloTTS-Chinese"
        model_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Download TTS checkpoint + config (skip if both already present)
        config_file = os.path.join(model_dir, "config.json")
        ckpt_file = os.path.join(model_dir, "checkpoint.pth")
        if os.path.isfile(config_file) and os.path.isfile(ckpt_file):
            print(f"\n  [{lang}] TTS 模型已存在，跳过下载: {model_dir}")
        else:
            print(f"\n  [{lang}] TTS 模型 ({repo_id}) -> {model_dir}")
            try:
                hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=model_dir)
                hf_hub_download(repo_id=repo_id, filename="checkpoint.pth", local_dir=model_dir)
                print(f"    ✓ checkpoint.pth + config.json")
            except Exception as e:
                print(f"    ✗ TTS 下载失败: {e}")

        # Download BERT models to ./models/{ModelName}/bert/{bert_id}/
        bert_models = LANG_TO_BERT.get(lang, [])
        for bert_id in bert_models:
            bert_save_name = bert_id.replace("/", "--")
            bert_save_dir = os.path.join(model_dir, "bert", bert_save_name)
            if bert_id in downloaded_bert:
                # Already handled in this run (shared across languages)
                continue
            # Skip if BERT already downloaded (config + weights present on disk)
            has_config = os.path.isfile(os.path.join(bert_save_dir, "config.json"))
            has_weights = os.path.isfile(os.path.join(bert_save_dir, "model.safetensors")) or \
                os.path.isfile(os.path.join(bert_save_dir, "pytorch_model.bin"))
            if has_config and has_weights:
                downloaded_bert.add(bert_id)
                print(f"  [{lang}] BERT ({bert_id}) 已存在，跳过下载: bert/{bert_save_name}/")
                continue
            downloaded_bert.add(bert_id)
            print(f"  [{lang}] BERT ({bert_id}) -> bert/{bert_save_name}/")
            try:
                os.makedirs(bert_save_dir, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(bert_id)
                model = AutoModelForMaskedLM.from_pretrained(bert_id)
                tokenizer.save_pretrained(bert_save_dir)
                model.save_pretrained(bert_save_dir)
                print(f"    ✓ {bert_id}")
            except Exception as e:
                print(f"    ✗ BERT 下载失败: {e}")


def download(lang=None, all_langs=False, skip_unidic=False, skip_models=False):
    """Download the models and data required to run MeloTTS.

    Args:
        lang: List of languages to download (e.g. ["ZH"], ["ZH", "EN"]); defaults to ["ZH"].
        all_langs: When True, download models for all languages.
        skip_unidic: When True, skip the unidic dictionary download.
        skip_models: When True, skip the model download (TTS + BERT).
    """
    if all_langs:
        languages = list(LANG_TO_HF_REPO.keys())
    else:
        languages = lang if lang else ["ZH"]

    print("MeloTTS 下载脚本")
    print(f"目标语言: {', '.join(languages)}")

    # 1. unidic (needed at MeloTTS import time regardless of language)
    if not skip_unidic:
        download_unidic()

    # 2. TTS + BERT models
    if not skip_models:
        download_models(languages)

    print("\n" + "=" * 60)
    print("下载完成！现在可以运行: python inference-torch.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MeloTTS 模型和数据集中下载")
    parser.add_argument(
        "--lang", nargs="+", default=["ZH"],
        help="要下载的语言列表，如: ZH EN JP FR ES KR EN_V2 EN_NEWEST (默认: ZH)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下载所有语言的模型"
    )
    parser.add_argument(
        "--skip-unidic", action="store_true",
        help="跳过 unidic 字典下载"
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="跳过模型下载（TTS + BERT）"
    )
    args = parser.parse_args()

    download(
        lang=args.lang,
        all_langs=args.all,
        skip_unidic=args.skip_unidic,
        skip_models=args.skip_models,
    )


if __name__ == "__main__":
    main()
