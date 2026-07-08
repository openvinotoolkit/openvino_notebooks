import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

from melo_torch.api import TTS

# Model directory layout: ./models/MeloTTS-Chinese/checkpoint.pth, config.json, bert/...
# Run download() first to fetch the models
LANG_TO_MODEL_NAME = {
    "EN": "MeloTTS-English",
    "EN_V2": "MeloTTS-English-v2",
    "EN_NEWEST": "MeloTTS-English-v3",
    "FR": "MeloTTS-French",
    "JP": "MeloTTS-Japanese",
    "ES": "MeloTTS-Spanish",
    "ZH": "MeloTTS-Chinese",
    "KR": "MeloTTS-Korean",
}

SYN_TEXT_BATCH = [
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "Hello! Welcome to the MeloTTS demo with OpenVINO acceleration.",
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "如果祖国需要，请把我埋在遥远的山岗，让我的身躯长成一道无形的屏障，往来的战友会为我泪落两行",
    "如果祖国需要，请让我紧握滚烫的钢枪，让我的双手握出一轮沧桑的红日，冰冷的大地会为我抚慰创伤",
    "如果祖国需要，请让更多的人走向杀敌的战场，让我和你的心房 在蓝天上，跳跃出永远不朽的乐章 。"
]


def run_inference(lang="ZH", speed=1.0, device="cpu", texts=None):
    """Load the PyTorch MeloTTS model and run inference with RTF benchmarking on a batch of texts.

    Args:
        lang: Language/model, e.g. "ZH", "EN" (default "ZH").
        speed: Speech speed (default 1.0).
        device: Inference device, e.g. "cpu" or "cuda:0" (default "cpu").
        texts: List of texts to synthesize; defaults to the built-in samples.
    """
    import soundfile as sf

    texts = texts if texts is not None else SYN_TEXT_BATCH

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models", LANG_TO_MODEL_NAME[lang])
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "checkpoint.pth")
    bert_dir = os.path.join(model_dir, "bert")

    # bert_dir: load the BERT model from local disk if present, otherwise download from HuggingFace
    bert_dir = bert_dir if os.path.isdir(bert_dir) else None

    model = TTS(language=lang, device=device, config_path=config_path,
                ckpt_path=ckpt_path, bert_dir=bert_dir)

    speaker_ids = model.hps.data.spk2id
    speaker_id = speaker_ids[lang]
    test_output_dir = "test_output_torch"

    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    # Warmup: warm up the model so first-run initialization overhead does not skew the RTF measurement
    print("Warmup...")
    model.tts_to_file("你好，世界。", speaker_id, os.path.join(test_output_dir, "warmup.wav"), speed=speed, quiet=True)
    print("Warmup 完成\n")

    total_audio_duration = 0.0
    total_infer_time = 0.0

    for i, text in enumerate(texts):
        output_path = os.path.join(test_output_dir, f'output_{i}.wav')

        start_time = time.time()
        model.tts_to_file(text, speaker_id, output_path, speed=speed, quiet=True)
        infer_time = time.time() - start_time

        # Read the duration of the generated audio
        audio_data, sr = sf.read(output_path)
        audio_duration = len(audio_data) / sr

        total_audio_duration += audio_duration
        total_infer_time += infer_time

        rtf = infer_time / audio_duration
        print(f"  [{i+1}/{len(texts)}] 推理: {infer_time:.2f}s | 音频: {audio_duration:.2f}s | RTF: {rtf:.3f}")

    print(f"\n{'='*50}")
    print(f"总推理时间: {total_infer_time:.2f}s")
    print(f"总音频时长: {total_audio_duration:.2f}s")
    print(f"平均实时率 (RTF): {total_infer_time / total_audio_duration:.3f}")
    print(f"  RTF < 1.0 表示比实时快, 值越小越快")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_inference()