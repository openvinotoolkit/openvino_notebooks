from api import OpenVoiceBaseClass
from mel_processing import spectrogram_torch
import torch
import librosa
import openvino as ov
import os
import re
import soundfile



def get_tts_forward(base_class: OpenVoiceBaseClass):
    for par in base_class.model.parameters():
        par.requires_grad = False
    
    speed = 1.0
    kwargs = dict(noise_scale = 0.667, length_scale = 1.0 / speed, noise_scale_w = 0.6, sdp_ratio = 0.2)

    def tts_forward_wrapper(x, x_lengths, sid):
        return base_class.model.infer(x, x_lengths, sid,
                                            noise_scale=kwargs['noise_scale'], 
                                            length_scale=kwargs['length_scale'], 
                                            noise_scale_w=kwargs['noise_scale_w'], 
                                            sdp_ratio=kwargs['sdp_ratio'])
    return tts_forward_wrapper

def get_converter_forward(base_class: OpenVoiceBaseClass):
    for par in base_class.model.parameters():
        par.requires_grad = False
    def converter_forward_wrapper(y, y_lengths, sid_src, sid_tgt):
        return base_class.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau=0.3)
    return converter_forward_wrapper


class OVOpenVoiceTTS(torch.nn.Module):
    def __init__(self, tts_model, noise_scale = 0.667, noise_scale_w = 0.6, speed = 1, sdp_ratio = 0.2, ir_path='openvoice_tts.xml'):
        super().__init__()
        self.tts_model = tts_model
        self.ir_path = ir_path

        self.default_kwargs = dict(
             noise_scale = noise_scale, 
             noise_scale_w = noise_scale_w,
             length_scale = 1 / speed,
             sdp_ratio = sdp_ratio
        )

    
    def forward(self, x, x_lengths, sid):
        for par in self.tts_model.model.parameters():
            par.requires_grad = False
        return self.tts_model.model.infer(x, x_lengths, sid, **self.default_kwargs)

    def get_example_input(self):
        stn_tst = self.tts_model.get_text('this is original text', self.tts_model.hps, False)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        speaker_id = torch.LongTensor([1])
        return (x_tst, x_tst_lengths, speaker_id)

    def compile(self, ov_device='CPU'):
        core = ov.Core()
        if os.path.exists(self.ir_path):
            self.ov_tts = core.read_model(self.ir_path)
        else:
            self.ov_tts = ov.convert_model(self, example_input=self.get_example_input())
        
        self.compiled_model = core.compile_model(self.ov_tts, ov_device)

    def tts(self, text, output_path, speaker, language='English', speed=1.0):
        tts_model = self.tts_model

        mark = tts_model.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = tts_model.split_sentences_into_pieces(text, mark)

        audio_list = []
        for t in texts:
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            t = f'[{mark}]{t}[{mark}]'
            stn_tst = tts_model.get_text(t, tts_model.hps, False)
            device = tts_model.device
            speaker_id = tts_model.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([speaker_id]).to(device)
                 # call OpenVino instead of torch
                audio = self.compiled_model(((x_tst, x_tst_lengths, sid)))[0][0, 0]
            audio_list.append(audio)
        audio = tts_model.audio_numpy_concat(audio_list, sr=tts_model.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, tts_model.hps.data.sampling_rate)

class OVOpenVoiceConvert(torch.nn.Module):
    def __init__(self, voice_conversion_model, tau=0.3, ir_path='openvoice_converter.xml'):
        super().__init__()
        self.voice_conversion_model = voice_conversion_model
        self.ir_path = ir_path
        
        self.default_kwargs = dict(
             tau = tau, 
        )
    
    def get_example_input():
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(*(1, 256, 1))
        source_se = torch.randn(*(1, 256, 1))
        return (y, y_lengths, source_se, target_se)
    
    def compile(self, ov_device='CPU'):
        core = ov.Core()
        if os.path.exists(self.ir_path):
            self.ov_voice_conversion = core.read_model(self.ir_path)
        else:
            self.ov_voice_conversion = ov.convert_model(self, example_input=self.get_example_input())
        
        self.compiled_model = core.compile_model(self.ov_voice_conversion, ov_device)
    
    def forward(self, y, y_lengths, sid_src, sid_tgt):
        for par in self.voice_conversion_model.model.parameters():
            par.requires_grad = False
        return self.voice_conversion_model.model.infer(y, y_lengths, sid_src, sid_tgt, **self.default_kwargs)
    
    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        model = self.voice_conversion_model

        hps = model.hps
        # load audio
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()
        
        with torch.no_grad():
            y = torch.FloatTensor(audio).to(model.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False).to(model.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(model.device)
            audio = model.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
                        0, 0].data.cpu().float().numpy()
            # call OpenVino instead of torch
            audio = self.compiled_model((spec, spec_lengths, src_se, tgt_se))[0][0, 0]
            audio = model.add_watermark(audio, message)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, hps.data.sampling_rate)
