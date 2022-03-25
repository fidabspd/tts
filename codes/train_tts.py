import os
import time
import math
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import *
from transformer_torch import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--configs_file_name', type=str, default='./hyperparams.json')

    parser.add_argument('--wav_path', type=str, default='../data/wav/')
    parser.add_argument('--speaker', type=str, default='여1')
    parser.add_argument('--script_file_name', type=str, default='../data/scripts.xlsx')
    parser.add_argument('--graph_log_path', type=str, default='../logs/graph/')
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='single_speaker_tts')
    parser.add_argument('--train_log_path', type=str, default='../logs/train_logs/')

    parser.add_argument('--text_seq_len', type=int, default=2000)
    parser.add_argument('--speech_seq_len', type=int, default=2000)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pf_dim', type=int, default=512)
    parser.add_argument('--dropout_ratio', type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--clip', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--es_patience', type=int, default=5)
    parser.add_argument('--validate', type=bool, default=False)

    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--frame_shift', type=int, default=0.0125)
    parser.add_argument('--frame_length', type=int, default=0.05)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mels', type=int, default=80)

    return parser.parse_args()


def create_tensorboard_graph(model, inputs, path):
    try:
        exist = bool(len(os.listdir(path)))
    except:
        exist = False
    if not exist:
        writer = SummaryWriter(path)
        writer.add_graph(model, inputs)
        writer.close()
        print('Saved model graph')
    else:
        print('graph already exists')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train_one_epoch(model, dl, optimizer, criterion, clip, device, n_check=5):

    n_data = len(dl.dataset)
    n_batch = len(dl)
    batch_size = dl.batch_size
    if n_check < 0:
        print('n_check must be larger than 0. Adjust `n_check = 0`')
        n_check = 0
    if n_batch < n_check:
        print(f'n_check should be smaller than n_batch. Adjust `n_check = {n_batch}`')
        n_check = n_batch
    if n_check:
        check = [int(n_batch/n_check*(i+1)) for i in range(n_check)]
    train_loss = 0

    model.train()
    for b, (inp, tar) in enumerate(dl):
        inp, tar = inp.to(device), tar.to(device)

        outputs, _ = model(inp, tar[:,:-1])

        output_dim = outputs.shape[-1]
        outputs = outputs.contiguous().view(-1, output_dim)
        tar = tar[:,1:].contiguous().view(-1)
        loss = criterion(outputs, tar)
        train_loss += loss.item()/n_data

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if n_check and b+1 in check:
            n_data_check = b*batch_size + len(inp)
            train_loss_check = train_loss*n_data/n_data_check
            print(f'loss: {train_loss_check:>10f}  [{n_data_check:>5d}/{n_data:>5d}]')

    return train_loss


def evaluate(model, dl, criterion, device):
    n_data = len(dl.dataset)
    
    valid_loss = 0

    model.eval()
    with torch.no_grad():
        for inp, tar in dl:
            inp, tar = inp.to(device), tar.to(device)
            outputs, _ = model(inp, tar[:,:-1])

            output_dim = outputs.shape[-1]

            outputs = outputs.contiguous().view(-1, output_dim)
            tar = tar[:,1:].contiguous().view(-1)
            loss = criterion(outputs, tar)

            valid_loss += loss.item()/n_data

    return valid_loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, n_epochs, es_patience, train_dl, valid_dl, optimizer,
          criterion, clip, device, model_path, train_log_path, model_name='chatbot'):
    if train_log_path is not None:
        writer = SummaryWriter(train_log_path)
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        print('-'*30, f'\nEpoch: {epoch+1:02}', sep='')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        if train_log_path is not None:
            writer.add_scalar('train loss', train_loss, epoch)
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion, device)
            if train_log_path is not None:
                writer.add_scalar('valid loss', valid_loss, epoch)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_dl is not None:
            if valid_loss < best_valid_loss:
                best_epoch = epoch
                print('Best!')
                best_valid_loss = valid_loss
                torch.save(model, model_path+model_name+'.pt')

        print(f'Train Loss: {train_loss:.3f}\nEpoch Time: {epoch_mins}m {epoch_secs}s')
        if valid_dl is not None:
            print(f'Validation Loss: {valid_loss:.3f}')

            if epoch-best_epoch >= es_patience:
                print(f'\nBest Epoch: {best_epoch+1:02}')
                print(f'\tBest Train Loss: {train_loss:.3f}')
                print(f'\tBest Validation Loss: {valid_loss:.3f}')
                break
    
    if train_log_path is not None:
        writer.close()
    if valid_dl is None:
        torch.save(model, model_path+model_name+'.pt')


def main(args):

    CONFIGS_FILE_NAME = args.configs_file_name
    with open(CONFIGS_FILE_NAME) as f:
        configs = json.load(f)
    args_dict = vars(args)
    args_dict = {key: args_dict[key] for key in args_dict.keys() if args_dict[key] is not None}
    configs.update(args_dict)


    WAV_PATH = configs['wav_path']
    SPEAKER = configs['speaker']
    SCRIPT_FILE_NAME = configs['script_file_name']
    GRAPH_LOG_PATH = configs['graph_log_path']
    MODEL_PATH = configs['model_path']
    MODEL_NAME = configs['model_name']
    TRAIN_LOG_PATH = configs['train_log_path']

    TEXT_SEQ_LEN = configs['text_seq_len']
    SPEECH_SEQ_LEN = configs['speech_seq_len']
    N_LAYERS = configs['n_layers']
    HIDDEN_DIM = configs['hidden_dim']
    N_HEADS = configs['n_heads']
    PF_DIM = configs['pf_dim']
    DROPOUT_RATIO = configs['dropout_ratio']

    SR = configs['sr']
    FRAME_SHIFT = configs['frame_shift']
    FRAME_LENGTH = configs['frame_length']
    N_FFT = configs['n_fft']
    N_MELS = configs['n_mels']
    HOP_LENGTH = int(SR*FRAME_SHIFT)
    WIN_LENGTH = int(SR*FRAME_LENGTH)

    BATCH_SIZE = configs['batch_size']
    LEARNING_RATE = configs['learning_rate']
    CLIP = configs['clip']
    N_EPOCHS = configs['n_epochs']
    ES_PATIENCE = configs['es_patience']
    VALIDATE = configs['validate']
    PAD_IDX = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Load data
    ds = get_single_speaker_dataset(
        SPEAKER, WAV_PATH, SCRIPT_FILE_NAME, SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH)
    train_dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Set model
    transformer = Transformer(
        82, N_MELS, N_LAYERS, HIDDEN_DIM, N_HEADS, PF_DIM,
        TEXT_SEQ_LEN, SPEECH_SEQ_LEN, PAD_IDX, DROPOUT_RATIO, device
    ).to(device)

    print(f'# of trainable parameters: {count_parameters(transformer):,}')
    transformer.apply(initialize_weights)

    inp, tar = iter(train_dl).next()
    inp, tar = inp.to(device), tar.to(device)
    create_tensorboard_graph(transformer, (inp, tar), GRAPH_LOG_PATH)

    # Train model
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_IDX)

    if not VALIDATE:
        valid_dl = None
    train(
        transformer, N_EPOCHS, ES_PATIENCE, train_dl, valid_dl,
        optimizer, criterion, CLIP, device, MODEL_PATH, TRAIN_LOG_PATH, MODEL_NAME
    )

    if VALIDATE:
        transformer = torch.load(MODEL_PATH+MODEL_NAME+'.pt')
        valid_loss = evaluate(transformer, valid_dl, criterion, device)
        print(f'Valid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):.3f}')


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    main(args)
