import json

HYPERPARMAS_FILE_PATH = './hyperparams.json'
with open(HYPERPARMAS_FILE_PATH) as f:
    configs = json.load(f)

SR = configs['sr']
FRAME_STRIDE = configs['frame_stride']
FRAME_LENGTH = configs['frame_length']
PREEMPHASIS = configs['preemphasis']
N_FFT = configs['n_fft']
N_MELS = configs['n_mels']
MAX_DB = configs['max_db']
REF_DB = configs['ref_db']


# For Text
PAD = '_'
SOS = '@'
EOS = '|'
PUNC = ['.', ',', '?', '!', '\'', '\"', '-', '~', '…']
SPACE = ' '

JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + [SPACE]
ALL_SYMBOLS = [PAD] + [SOS] + [EOS] + VALID_CHARS

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}
