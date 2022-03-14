import os

import numpy as np
import pandas as pd

import re
from jamo import hangul_to_jamo

import librosa



##############
# For Script
##############

def get_data_list(speaker, wav_path):
    pattern = re.compile(f'^{speaker}_')
    data_list = [folder for folder in os.listdir(wav_path) if pattern.match(folder)]
    return data_list

def load_script(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)['대사']



##############
# For Audio
##############

def get_mel(fpath, sr, n_mels, n_fft, hop_length, win_length):

    y, _ = librosa.load(fpath, sr=sr)
    y, _ = librosa.effects.trim(y)
    
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    mel = librosa.feature.melspectrogram(
        y=y, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length)
    
    return mel.T.astype(np.float32)
    