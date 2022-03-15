import os

import numpy as np
import pandas as pd

import re
from jamo import hangul_to_jamo

import librosa

from torch.utils.data import Dataset, ConcatDataset, DataLoader



##############
# For Script
##############

def get_data_list(speaker, wav_path):
    pattern = re.compile(f'^{speaker}_')
    data_list = [folder for folder in os.listdir(wav_path) if pattern.match(folder)]
    return data_list

def load_script(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)['대사']


PAD = '_'
SOS = '@'
EOS = '|'
PUNC = ['!', '\'', '\"', '(', ')', ',', '-', '.', ':', ';', '?']
SPACE = ' '

JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + [SPACE]
ALL_SYMBOLS = [PAD] + [SOS] + [EOS] + VALID_CHARS

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}

def tokenize(text, as_id=False):
    tokens = list(hangul_to_jamo(text))

    if as_id:
        return [char_to_id[SOS]] + [char_to_id[token] for token in tokens] + [char_to_id[EOS]]
    else:
        return [SOS] + [token for token in tokens] + [EOS]



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
    


# Dataset
class TextMelDataset(Dataset):
    
    def __init__(self, scripts, audio_path, sr, n_mels, n_fft, hop_length, win_length):
        self.scripts = scripts
        self.audio_path = audio_path
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def __len__(self):
        return min(
            len(self.scripts),
            len(os.listdir(self.audio_path))
        )
    
    def __getitem__(self, idx):
        # script
        script = self.scripts[idx]
        # tokens = tokenize(script, as_id=True)
        
        # audio
        fpath = os.path.join(self.audio_path, str(idx+1)+'.wav')
        mel = get_mel(fpath, self.sr, self.n_mels, self.n_fft, self.hop_length, self.win_length)
        
        return script, mel


def get_single_speaker_dataset(speaker, wav_path, script_path, sr, n_mels, n_fft, hop_length, win_length):
    data_list = get_data_list(speaker, wav_path)

    concat_dataset = []
    print(f'Loading {data_list} ...')
    for sheet_name in data_list:
        sheet_name = sheet_name.split('_')[1]
        
        script = load_script(script_path, sheet_name)
        audio_path = os.path.join(wav_path, speaker+'_'+sheet_name)
        
        text_mel_dataset = TextMelDataset(script, audio_path, sr, n_mels, n_fft, hop_length, win_length)
        concat_dataset.append(text_mel_dataset)
        print(f'{sheet_name} Done!')
        
    return ConcatDataset(concat_dataset)