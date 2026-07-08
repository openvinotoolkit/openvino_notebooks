import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

from melo_openvino.api import TTS

SYN_TEXT_BATCH = [
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "Hello! Welcome to the MeloTTS demo with OpenVINO acceleration.",
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    "如果祖国需要，请把我埋在遥远的山岗，让我的身躯长成一道无形的屏障，往来的战友会为我泪落两行",
    "如果祖国需要，请让我紧握滚烫的钢枪，让我的双手握出一轮沧桑的红日，冰冷的大地会为我抚慰创伤",
    "如果祖国需要，请让更多的人走向杀敌的战场，让我和你的心房 在蓝天上，跳跃出永远不朽的乐章 。"
]


def run_inference(lang="ZH", speed=1.0, device="cpu", ov_model_dir=None, texts=None):
    """Load the OpenVINO MeloTTS model and run inference with RTF benchmarking on a batch of texts.

    OpenVINO model directory layout:
        ./models/OpenVINO/MeloTTS-Chinese/
            melotts_enc.xml/bin, melotts_dec.xml/bin  (OV IR)
            config.json                               (model config)
            bert/                                     (BERT text frontend)
    Run download() and model_convert.convert(...) first to generate the files above.

    Args:
        lang: Language/model, e.g. "ZH", "EN" (default "ZH").
        speed: Speech speed (default 1.0).
        device: Inference device, e.g. "cpu" or "cuda:0" (default "cpu").
        ov_model_dir: OV IR directory, defaults to ./models/OpenVINO/MeloTTS-Chinese.
        texts: List of texts to synthesize; defaults to the built-in samples.
    """
    import soundfile as sf

    texts = texts if texts is not None else SYN_TEXT_BATCH

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if ov_model_dir is None:
        ov_model_dir = os.path.join(base_dir, "models", "OpenVINO", "MeloTTS-Chinese")

    model = TTS(language=lang, device=device, ov_model_dir=ov_model_dir)

    speaker_ids = model.hps.data.spk2id
    speaker_id = speaker_ids[lang]
    test_output_dir = "test_output_ov"

    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    # Warmup: warm up the model so first-run initialization overhead does not skew the RTF measurement
    print("Warmup...")
    model.tts_to_file("你好，世界。", speaker_id, os.path.join(test_output_dir, "warmup.wav"), speed=speed, quiet=True)
    print("Warmup finished!\n")

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
    print(f"Total inference time: {total_infer_time:.2f}s")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Average RTF: {total_infer_time / total_audio_duration:.3f}")
    print(f"  RTF < 1.0 means faster than real time; lower is better")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_inference()