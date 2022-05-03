import os
import collections
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import configs as cf
from preprocess import *
from network import *


class TextMelDataset(Dataset):
    
    def __init__(self, script_path, sheet_name, audio_path, sr, n_mels, n_fft,
                 hop_length, win_length):
        self.scripts = load_script(script_path, sheet_name)
        self.audio_path = audio_path
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.audio_file_list = os.listdir(self.audio_path)
    
    def __len__(self):
        return len(self.audio_file_list)
    
    def __getitem__(self, idx):
        file_name = self.audio_file_list[idx]
        file_num = int(file_name[:-4])

        # script
        script = self.scripts.query(f'Index == {file_num}')['대사'].tolist()[0]
        script = normalize_text(script)
        text_tokens = tokenize(script, as_id=True)
        
        # audio
        fpath = os.path.join(self.audio_path, str(file_num)+'.wav')
        mel = get_mel(fpath, self.sr, self.n_mels, self.n_fft, self.hop_length, self.win_length)
        mel = np.concatenate([
                np.zeros([1, self.n_mels], np.float32),
                mel,
                np.zeros([2, self.n_mels], np.float32)
        ], axis=0)  # <sos> + mel + <eos>
        
        return {
            'text_tokens': text_tokens, 'mel': mel,
            'text_tokens_len': len(text_tokens), 'mel_len': len(mel)
        }


def get_single_speaker_dataset(speaker, wav_path, script_path,
                               sr, n_mels, n_fft, hop_length, win_length):

    data_list = get_data_list(speaker, wav_path)

    concat_dataset = []
    print(f'Loading {data_list} ...')
    for sheet_name in data_list:
        sheet_name = sheet_name.split('_')[1]
        
        audio_path = os.path.join(wav_path, speaker+'_'+sheet_name)
        
        text_mel_dataset = TextMelDataset(
            script_path, sheet_name, audio_path, sr, n_mels, n_fft,
            hop_length, win_length)
        concat_dataset.append(text_mel_dataset)
        print(f'{sheet_name} Done!')
        
    return ConcatDataset(concat_dataset)


def collate_fn(batch):

    def pad_tokens(text_tokens, max_len):
        len_tokens = len(text_tokens)
        return np.pad(text_tokens, (0, max_len-len_tokens))

    def pad_mel(mel, max_len):
        len_mel = len(mel)
        return np.pad(mel, ((0, max_len-len_mel), (0, 0)))

    max_text_len = max([data['text_tokens_len'] for data in batch])
    max_speech_len = max([data['mel_len'] for data in batch])

    text_tokens = np.stack([
        pad_tokens(data['text_tokens'], max_text_len)
        for data in batch
    ])
    mel = np.stack([
        pad_mel(data['mel'], max_speech_len)
        for data in batch
    ])

    return {
        'text_tokens': torch.LongTensor(text_tokens),
        'mel': torch.FloatTensor(mel)
    }


def get_dl_by_ds(ds, batch_size, num_workers, shuffle=False):
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers)
    return dl


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
        nn.init.kaiming_uniform_(m.weight.data)


def tensor_dict_to_device(tensor_dict, device):
    
    assert isinstance(tensor_dict, collections.abc.Mapping),\
        f'tensor_dict is not dicts. Found {type(tensor_dict)}.'
    
    for k, v in tensor_dict.items():
        if isinstance(v, collections.abc.Mapping):
            tensor_dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            tensor_dict[k] = v.to(device)
        else:
            raise Exception(f'value of dict is not torch.Tensor. Found {type(v)}')


def evaluate(model, dl, criterion, device):

    n_data = len(dl.dataset)
    valid_loss = 0
    n_processed_data = 0

    model.eval()
    pbar = tqdm(dl)
    with torch.no_grad():
        for batch in pbar:
            tensor_dict_to_device(batch, device)
            text_tokens, mel = batch['text_tokens'], batch['mel']
            n_processed_data += len(mel)
            
            mel_pred, _, _, _, _, _, _, _ = model(text_tokens, mel[:,:-1])
            mel_pred = mel_pred.contiguous().view(-1)
            mel = mel[:,1:].contiguous().view(-1)
            loss = criterion(mel_pred, mel)
            valid_loss += loss.item()/n_data

            valid_loss_tmp = valid_loss*n_data/n_processed_data
            pbar.set_description(
                f'Valid Loss: {valid_loss_tmp:9.6f} | {n_processed_data}/{n_data} ')

    return valid_loss


def train_model(model, train_dl, valid_dl, optimizer, criterion, n_epochs,
                es_patience, clip, model_file_path, train_log_path, device):

    def train_one_epoch(model, dl, optimizer, criterion, clip, device):

        nonlocal writer
        nonlocal global_step

        n_data = len(dl.dataset)
        train_loss = 0
        n_processed_data = 0

        model.train()
        pbar = tqdm(dl)
        for batch in pbar:
            tensor_dict_to_device(batch, device)
            text_tokens, mel = batch['text_tokens'], batch['mel']
            now_batch_len = len(mel)
            n_processed_data += now_batch_len

            mel_pred, _, _, _, _, _, _, _ = model(text_tokens, mel[:,:-1])
            mel_pred = mel_pred.contiguous().view(-1)
            mel = mel[:,1:].contiguous().view(-1)
            loss = criterion(mel_pred, mel)
            train_loss += loss.item()/n_data

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss_tmp = train_loss*n_data/n_processed_data
            pbar.set_description(
                f'Train Loss: {train_loss_tmp:9.6f} | {n_processed_data}/{n_data} ')

            batch_loss = loss.item()/now_batch_len
            writer.add_scalars(
                'loss', {
                    'batch': batch_loss,
                }, global_step
            )
            
            global_step += 1

        return train_loss


    writer = SummaryWriter(train_log_path)
    global_step = 0

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):

        print(f'\nEpoch: {epoch+1}/{n_epochs}')
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, clip, device)
        valid_loss = evaluate(model, valid_dl, criterion, device)
        writer.add_scalars(
            'loss', {
                'train': train_loss,
                'valid': valid_loss,
            }, global_step
        )

        # Best model
        if valid_loss < best_valid_loss:
            print('Best!')
            best_epoch = epoch
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            torch.save(model, model_file_path)

        # Check point
        if (epoch+1)%10 == 0:
            torch.save(model, f'./model/single_speaker_tts_checkpoint_{epoch+1}.pt')

        # # Early stop
        # if epoch-best_epoch >= es_patience:
        #     break
    
    print(f'\nBest Epoch: {best_epoch+1:02}')
    print(f'\tBest Train Loss: {best_train_loss:.3f}')
    print(f'\tBest Valid Loss: {best_valid_loss:.3f}')

    writer.close()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    # Set Dataset
    full_ds = get_single_speaker_dataset(
        cf.SPEAKER, cf.WAV_PATH, cf.SCRIPT_FILE_NAME,
        cf.SR, cf.N_MELS, cf.N_FFT, cf.HOP_LENGTH, cf.WIN_LENGTH
    )

    # Split train, valid
    data_length = len(full_ds)
    train_ds_len = int(data_length*cf.TRAINSET_RATIO)
    valid_ds_len = data_length - train_ds_len
    train_ds, valid_ds = random_split(full_ds, [train_ds_len, valid_ds_len])

    # Set DataLoader
    train_dl = get_dl_by_ds(train_ds, cf.BATCH_SIZE, cf.DL_NUM_WORKERS, shuffle=True)
    valid_dl = get_dl_by_ds(valid_ds, cf.BATCH_SIZE, cf.DL_NUM_WORKERS, shuffle=False)

    # Set model
    transformer = Transformer(
        len(cf.ALL_SYMBOLS), cf.N_MELS, cf.N_LAYERS, cf.HIDDEN_DIM,
        cf.N_HEADS, cf.PF_DIM, cf.TEXT_SEQ_LEN, cf.SPEECH_SEQ_LEN,
        cf.PAD_IDX, cf.DROPOUT_RATIO, device
    ).to(device)
    print(f'# of trainable parameters: {count_parameters(transformer):,}')
    transformer.apply(initialize_weights)

    # Tensorboard graph file
    sample_data = iter(train_dl).next()
    tensor_dict_to_device(sample_data, device)
    text_tokens, mel = sample_data['text_tokens'], sample_data['mel']
    create_tensorboard_graph(transformer, (text_tokens, mel[:,:-1]), cf.GRAPH_LOG_PATH)

    # Train model
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cf.LEARNING_RATE)
    criterion = nn.L1Loss(reduction='sum')
    train_model(
        transformer, train_dl, valid_dl, optimizer, criterion, cf.N_EPOCHS,
        cf.ES_PATIENCE, cf.CLIP, cf.MODEL_FILE_PATH, cf.TRAIN_LOG_PATH, device
    )



if __name__ == '__main__':

    main()
