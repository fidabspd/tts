import json

HYPERPARMAS_FILE_PATH = './hyperparams.json'
with open(HYPERPARMAS_FILE_PATH) as f:
    configs = json.load(f)

WAV_PATH = configs['wav_path']
SPEAKER = configs['speaker']
SCRIPT_FILE_NAME = configs['script_file_name']
GRAPH_LOG_PATH = configs['graph_log_path']
MODEL_FILE_PATH = configs['model_file_path']
TRAIN_LOG_PATH = configs['train_log_path']

TEXT_SEQ_LEN = configs['text_seq_len']
SPEECH_SEQ_LEN = configs['speech_seq_len']
N_LAYERS = configs['n_layers']
HIDDEN_DIM = configs['hidden_dim']
N_HEADS = configs['n_heads']
PF_DIM = configs['pf_dim']
DROPOUT_RATIO = configs['dropout_ratio']

REF_DB = configs['ref_db']
MAX_DB = configs['max_db']

SR = configs['sr']
FRAME_SHIFT = configs['frame_shift']
FRAME_LENGTH = configs['frame_length']
N_FFT = configs['n_fft']
N_MELS = configs['n_mels']
HOP_LENGTH = int(SR*FRAME_SHIFT)
WIN_LENGTH = int(SR*FRAME_LENGTH)

TRAINSET_RATIO = configs['trainset_ratio']
DL_NUM_WORKERS = configs['dl_num_workers']
BATCH_SIZE = configs['batch_size']
LEARNING_RATE = configs['learning_rate']
CLIP = configs['clip']
N_EPOCHS = configs['n_epochs']
ES_PATIENCE = configs['es_patience']
PAD_IDX = 0



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
