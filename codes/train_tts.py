import os
import math
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs import *
from preprocess import *
from transformer_torch import *


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--configs_file_name', type=str, default='./hyperparams.json')

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


# def train_one_epoch(model, dl, optimizer, criterion, clip, device):

#     n_data = len(dl.dataset)
#     train_loss = 0
#     n_processed_data = 0

#     model.train()
#     pbar = tqdm(dl)
#     for batch in pbar:
#         tensor_dict_to_device(batch, device)
#         inputs, targets = batch['inputs'], batch['targets']
#         n_processed_data += len(targets)

#         pred = model(**inputs)
#         loss = criterion(pred, targets)
#         train_loss += loss.item()/n_data

#         optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()

#         train_loss_tmp = train_loss*n_data/n_processed_data
#         pbar.set_description(
#             f'Train Loss: {train_loss_tmp:9.6f} | {n_processed_data:6d}/{n_data:6d} ')

#     return train_loss


def train_one_epoch(model, dl, optimizer, criterion, clip, device):

    n_data = len(dl.dataset)
    train_loss = 0; n_processed_data = 0
    model.train()
    pbar = tqdm(dl)
    for inp, tar in pbar:
        n_processed_data += len(inp)
        inp, tar = inp.to(device), tar.to(device)

        outputs, _, _ = model(inp, tar[:,:-1])

        outputs = outputs.contiguous().view(-1)
        tar = tar[:,1:].contiguous().view(-1)
        loss = criterion(outputs, tar)
        train_loss += loss.item()/n_data

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss_tmp = train_loss*n_data/n_processed_data
        pbar.set_description(
            f'{n_processed_data:4d}/{n_data:4d} | loss: {train_loss_tmp:10.6f}')

    return train_loss


# def evaluate(model, dl, criterion, device):

#     n_data = len(dl.dataset)
#     valid_loss = 0
#     n_processed_data = 0

#     model.eval()
#     pbar = tqdm(dl)
#     with torch.no_grad():
#         for batch in pbar:
#             tensor_dict_to_device(batch, device)
#             inputs, targets = batch['inputs'], batch['targets']
#             n_processed_data += len(targets)
            
#             pred = model(**inputs)
#             loss = criterion(pred, targets)
#             valid_loss += loss.item()/n_data

#             valid_loss_tmp = valid_loss*n_data/n_processed_data
#             pbar.set_description(
#                 f'Valid Loss: {valid_loss_tmp:9.6f} | {n_processed_data:6d}/{n_data:6d} ')

#     return valid_loss


def evaluate(model, dl, criterion, device):
    n_data = len(dl.dataset)
    
    valid_loss = 0

    model.eval()
    with torch.no_grad():
        for inp, tar in dl:
            inp, tar = inp.to(device), tar.to(device)
            outputs, _, _ = model(inp, tar[:,:-1])

            output_dim = outputs.shape[-1]

            outputs = outputs.contiguous().view(-1, output_dim)
            tar = tar[:,1:].contiguous().view(-1)
            loss = criterion(outputs, tar)

            valid_loss += loss.item()/n_data

    return valid_loss


# def train_model(model, train_dl, valid_dl, optimizer, criterion, n_epochs,
#                 es_patience, clip, model_file_path, train_log_path, device):
#     if train_log_path is not None:
#         writer = SummaryWriter(train_log_path)
#     best_train_loss = float('inf')
#     best_valid_loss = float('inf')
#     best_epoch = 0

#     for epoch in range(n_epochs):

#         print(f'Epoch: {epoch+1}/{n_epochs}')
#         train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
#         valid_loss = evaluate(model, valid_dl, criterion, device)
#         if train_log_path is not None:
#             writer.add_scalars('loss', {
#                     'train_loss':train_loss,
#                     'valid_loss':valid_loss,
#                 }, epoch+1)

#         if valid_loss < best_valid_loss:
#             print('Best!\n')
#             best_epoch = epoch
#             best_train_loss = train_loss
#             best_valid_loss = valid_loss
#             torch.save(model, model_file_path)

#         if epoch-best_epoch >= es_patience:
#             print(f'\nBest Epoch: {best_epoch+1:02}')
#             print(f'\tBest Train Loss: {best_train_loss:.3f}')
#             print(f'\tBest Validation Loss: {best_valid_loss:.3f}')
#             break
    
#     if train_log_path is not None:
#         writer.close()


def train(model, n_epochs, es_patience, train_dl, valid_dl, optimizer,
          criterion, clip, device, model_path, train_log_path, model_name='chatbot'):
    if train_log_path is not None:
        writer = SummaryWriter(train_log_path)
    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):

        print(f'Epoch: {epoch+1}/{n_epochs}')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        if train_log_path is not None:
            writer.add_scalar('train loss', train_loss, epoch)
        if valid_dl is not None:
            valid_loss = evaluate(model, valid_dl, criterion, device)
            if train_log_path is not None:
                writer.add_scalar('valid loss', valid_loss, epoch)


        if valid_dl is not None:
            if valid_loss < best_valid_loss:
                best_epoch = epoch
                print('Best!')
                best_valid_loss = valid_loss
                torch.save(model, model_path+model_name+'.pt')
        else:
            if train_loss < best_train_loss:
                best_epoch = epoch
                print('Best!')
                best_train_loss = train_loss
                torch.save(model, model_path+model_name+'.pt')

        if valid_dl is not None:
            print(f'Valid Loss: {valid_loss:.3f}')

        if epoch-best_epoch >= es_patience:
            print(f'\nBest Epoch: {best_epoch+1:02}')
            print(f'\tBest Train Loss: {train_loss:.3f}')
            print(f'\tBest Validation Loss: {valid_loss:.3f}')
            break
    
    if train_log_path is not None:
        writer.close()


def main(args):

    # CONFIGS_FILE_NAME = args.configs_file_name
    # with open(CONFIGS_FILE_NAME) as f:
    #     configs = json.load(f)
    # args_dict = vars(args)
    # args_dict = {key: args_dict[key] for key in args_dict.keys() if args_dict[key] is not None}
    # configs.update(args_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Load data
    ds = get_single_speaker_dataset(
        SPEAKER, WAV_PATH, SCRIPT_FILE_NAME, SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH)
    train_dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Set model
    transformer = Transformer(
        len(ALL_SYMBOLS), N_MELS, N_LAYERS, HIDDEN_DIM, N_HEADS, PF_DIM,
        TEXT_SEQ_LEN, SPEECH_SEQ_LEN, PAD_IDX, DROPOUT_RATIO, device
    ).to(device)

    print(f'# of trainable parameters: {count_parameters(transformer):,}')
    transformer.apply(initialize_weights)

    inp, tar = iter(train_dl).next()
    inp, tar = inp.to(device), tar.to(device)
    create_tensorboard_graph(transformer, (inp, tar), GRAPH_LOG_PATH)

    # Train model
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss(reduction='sum')

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
