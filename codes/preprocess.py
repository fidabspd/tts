import os
import re
import numpy as np
import pandas as pd
from jamo import hangul_to_jamo
import librosa

import configs as cf



##############
# For Script
##############

def get_data_list(speaker, wav_path):
    pattern = re.compile(f'^{speaker}_')
    data_list = [folder for folder in os.listdir(wav_path) if pattern.match(folder) and '어학' not in folder]
    return data_list

def load_script(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)


def tokenize(text, as_id=False):
    tokens = list(hangul_to_jamo(text))

    if as_id:
        return [cf.char_to_id[cf.SOS]] + [cf.char_to_id[token] for token in tokens] + [cf.char_to_id[cf.EOS]]
    else:
        return [cf.SOS] + [token for token in tokens] + [cf.EOS]


def normalize_en(text):
    
    en_to_kor_dict = {
            'A': '에이', 'a': '에이',
            'B': '비', 'b': '비',
            'C': '씨', 'c': '씨',
            'D': '디', 'd': '디',
            'E': '이', 'e': '이',
            'F': '에프', 'f': '에프',
            'G': '지', 'g': '지',
            'H': '에이치', 'h': '에이치',
            'I': '아이', 'i': '아이',
            'J': '제이', 'j': '제이',
            'K': '케이', 'k': '케이',
            'L': '엘', 'l': '엘',
            'M': '엠', 'm': '엠',
            'N': '엔', 'n': '엔',
            'O': '오', 'o': '오',
            'P': '피', 'p': '피',
            'Q': '큐', 'q': '큐',
            'R': '알', 'r': '알',
            'S': '에스', 's': '에스',
            'T': '티', 't': '티',
            'U': '유', 'u': '유',
            'V': '브이', 'v': '브이',
            'W': '더블유', 'w': '더블유',
            'X': '엑스', 'x': '엑스',
            'Y': '와이', 'y': '와이',
            'Z': '지', 'z': '지',
    }

    return re.sub('[A-Za-z]', lambda x: en_to_kor_dict[x.group()], text)


def int_to_kor_under_4d(num):
    
    num = str(num)
    if num in ['0', '00', '000', '0000']:
        return ''
    num_to_kor_list = [''] + list('일이삼사오육칠팔구')
    unit_to_kor_list = [''] + list('십백천')
    
    result = ''
    i = len(num)
    for n in num:
        i -= 1
        n = int(n)
        str_unit = unit_to_kor_list[i]
        if str_unit != '' and n == 1:
            str_n = ''
        else:
            str_n = num_to_kor_list[n]
        result += str_n+str_unit
    return result


def int_to_kor(num):
    
    result = ''
    
    unit_to_kor_list = [''] + list('만억조경해')
    
    num = str(num)
    n_digit = len(num)
    i, d = n_digit%4, n_digit//4
    if i:
        result += int_to_kor_under_4d(num[:i])+unit_to_kor_list[d]
    d -= 1
    while i+4 <= n_digit:
        n = num[i:i+4]
        result += int_to_kor_under_4d(n)+unit_to_kor_list[d]
        i += 4
        d -= 1
        
    if result[:2] == '일만':
        result = result[1:]
        
    return result


def num_to_kor(num):
    
    num = str(num)
    if num == '0':
        return '영'
    num_to_kor_list = list('영일이삼사오육칠팔구')

    if '.' in num:
        _int, _float = num.split('.')
        float_to_kor_result = ''
        for f in _float:
            float_to_kor_result += num_to_kor_list[int(f)]
        float_to_kor_result = '쩜'+float_to_kor_result
    else:
        _int = num
        float_to_kor_result = ''
    assert len(_int) <= 24, 'Too long number'
    
    int_to_kor_result = int_to_kor(_int)
    
    return int_to_kor_result+float_to_kor_result


def normalize_num(text):
    return re.sub('\d+[.]{0,1}\d*', lambda x: num_to_kor(x.group()), text)


def normalize_quote(text):
    return re.sub('[<>‘’〈〉『』]', '\'', text)


def normalize_etc(text):
    text = re.sub('%', '퍼센트', text)
    text = re.sub('to', '투', text)
    text = re.sub('[·”]', ' ', text)
    return text


def normalize_text(text):
    text = normalize_etc(text)
    text = normalize_quote(text)
    text = normalize_en(text)
    text = normalize_num(text)
    return text



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
