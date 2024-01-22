def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default", ov_model = None):
    hps = self.hps
    # load audio
    audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
    audio = torch.tensor(audio).float()
    
    with torch.no_grad():
        y = torch.FloatTensor(audio).to(self.device)
        y = y.unsqueeze(0)
        spec = spectrogram_torch(y, hps.data.filter_length,
                                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                center=False).to(self.device)
        spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
        audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
                    0, 0].data.cpu().float().numpy()
        # call OpenVino instead of torch
        if ov_model is None:
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
            0, 0].data.cpu().float().numpy()
        else:
            audio = ov_model((spec, spec_lengths, src_se, tgt_se))[0][0, 0]
        audio = self.add_watermark(audio, message)
        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, hps.data.sampling_rate)

def tts(self, text, output_path, speaker, language='English', speed=1.0, ov_model=None):
    mark = self.language_marks.get(language.lower(), None)
    assert mark is not None, f"language {language} is not supported"

    texts = self.split_sentences_into_pieces(text, mark)

    audio_list = []
    for t in texts:
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = f'[{mark}]{t}[{mark}]'
        stn_tst = self.get_text(t, self.hps, False)
        device = self.device
        speaker_id = self.hps.speakers[speaker]
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            sid = torch.LongTensor([speaker_id]).to(device)
            if ov_model is None:
                audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.6,
                                    length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
            else:
                audio = ov_model(((x_tst, x_tst_lengths, sid)))[0][0, 0]
        audio_list.append(audio)
    audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

    if output_path is None:
        return audio
    else:
        soundfile.write(output_path, audio, self.hps.data.sampling_rate)