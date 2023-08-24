from pathlib import Path
import pickle
import numpy as np
import torch


with open('calibration/librispeech_asr_dummy_last_15.pkl', 'rb') as f:
    encoder_init_data_last, decoder_init_data_last = pickle.load(f)

with open('calibration/librispeech_asr_dummy_reversed_15.pkl', 'rb') as f:
    encoder_init_data_rev, decoder_init_data_rev = pickle.load(f)
    encoder_init_data_rev, decoder_init_data_rev = list(reversed(encoder_init_data_rev)), list(reversed(decoder_init_data_rev))

for ls, rs in zip(encoder_init_data_last, encoder_init_data_rev):
    for l, r in zip(ls, rs):
        diff = torch.abs(l - r).max()
        if diff > 1e-6:
            print(diff)


for ls, rs in zip(decoder_init_data_last, decoder_init_data_rev):
    for l, r in zip(ls, rs):
        for k in l.keys():
            if np.prod(l[k].shape) == np.prod(r[k].shape) == 0:
                continue
            diff = np.abs(l[k] - r[k]).max()
            if diff > 1e-6:
                print(k, diff)
