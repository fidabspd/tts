import torch
import soundfile as sf
import matplotlib.pyplot as plt

import configs as cf
from preprocess import *



class TextToSpeech():

    def __init__(self, transformer, device):
        self.transformer = transformer.to(device)
        self.transformer.eval()
        self.device = device

    def tts(self, text):
        
        text_tokens = tokenize(normalize_text('@'+text+'|'), as_id=True)
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)
        text_tokens_mask = self.transformer.create_padding_mask(text_tokens)
        with torch.no_grad():
            text_encd = self.transformer.encoder(text_tokens, text_tokens_mask)

        mel_sos = torch.zeros([1, cf.N_MELS]).unsqueeze(0).to(self.device)
        for i in range(self.transformer.speech_seq_len):
            if i == 0:
                mel_input = mel_sos
            else:
                mel_input = torch.concat((mel_sos, mel_pred[:, :i+1, :]), axis=1)
            mel_mask = self.transformer.create_padding_mask(mel_input, True)
            with torch.no_grad():
                mel_pred, _, attention = self.transformer.decoder(
                    mel_input, text_encd, mel_mask, text_tokens_mask)
        
        self.text = text
        self.text_tokens = text_tokens.detach().cpu().numpy()
        self.mel_pred = mel_pred[0].detach().cpu().numpy()
        self.attention = attention
        self.generated = True
        
        return mel_pred, attention
    
    def save_tts_result(self, save_file_path):
        if not self.generated:
            raise Exception('Call `tts` first')
        inversed = librosa.feature.inverse.mel_to_audio(
            self.mel_pred.T, sr=cf.SR, hop_length=cf.HOP_LENGTH, win_length=cf.WIN_LENGTH)
        
        sf.write(save_file_path, inversed, cf.SR)
        print(f'Saved tts result!\npath: {save_file_path}')

    def plot_attention_weights(self, draw_mean=False):
        if not self.generated:
            raise Exception('Call `tts` first')

        attention = self.attention.squeeze(0)
        if draw_mean:
            attention = torch.mean(attention, dim=0, keepdim=True)
        attention = attention.cpu().detach().numpy()

        n_col = 4
        n_row = (attention.shape[0]-1)//n_col + 1
        fig = plt.figure(figsize = (n_col*6, n_row*6))
        for i in range(attention.shape[0]):
            plt.subplot(n_row, n_col, i+1)
            plt.matshow(attention[i], fignum=False)
            plt.xticks(range(len(self.text_tokens)), self.text_tokens)
        plt.show()
