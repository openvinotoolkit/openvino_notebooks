import os
import re
import json
import soundfile
import numpy as np
from tqdm import tqdm

from . import utils
from .split_utils import split_sentence
from .download_utils import load_or_download_config
from .openvino_infer import OVSynthesizer

class TTS:
    def __init__(self, 
                language,
                device='cpu',
                use_hf=True,
                config_path=None,
                ckpt_path=None,
                bert_dir=None,
                ov_model_dir=None,
                ov_device=None):
        # OpenVINO 推理设备：cpu->CPU, cuda->GPU
        if ov_device is None:
            ov_device = 'GPU' if 'cuda' in str(device) else 'CPU'

        if ov_model_dir is None:
            raise ValueError("必须提供 ov_model_dir 以定位 OpenVINO 模型目录")

        # config.json: 优先使用显式参数，否则从 ov_model_dir 中加载
        if config_path is None:
            config_path = os.path.join(ov_model_dir, 'config.json')
        # bert_dir: 优先使用显式参数，否则从 ov_model_dir/bert 加载
        if bert_dir is None:
            _bert = os.path.join(ov_model_dir, 'bert')
            if os.path.isdir(_bert):
                bert_dir = _bert

        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)
        symbols = hps.symbols

        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.bert_dir = bert_dir

        self.model = OVSynthesizer(ov_model_dir, device=ov_device)

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = "CPU"
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id, bert_dir=self.bert_dir)
            # get_text_for_tts_infer 现在返回 numpy 数组
            x_tst = phones[np.newaxis, :]           # [1, T]
            tones = tones[np.newaxis, :]             # [1, T]
            lang_ids = lang_ids[np.newaxis, :]       # [1, T]
            bert = bert[np.newaxis, :]               # [1, D, T]
            ja_bert = ja_bert[np.newaxis, :]         # [1, D, T]
            x_tst_lengths = np.array([phones.shape[0]], dtype=np.int64)
            speakers = np.array([speaker_id], dtype=np.int64)
            audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1. / speed,
                )[0, 0]
            audio_list.append(audio)
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
