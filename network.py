import torch
from torch import nn

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0, f'hidden_dim must be multiple of n_heads.'
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim//n_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        # in_shape: [batch_size, seq_len, hidden_dim]
        # out_shape: [batch_size, seq_len, hidden_dim]
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.device = device

    def split_heads(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, self.n_heads, self.head_dim)
        # [batch_size, seq_len, n_heads, head_dim]
        splits = inputs.permute(0, 2, 1, 3)
        return splits  # [batch_size, n_heads, seq_len, head_dim]

    def scaled_dot_product_attention(self, query, key, value, mask):
        key_t = key.permute(0, 1, 3, 2)
        energy = torch.matmul(query, key_t) / self.scale  # [batch_size, n_heads, query_len, key_len]
        # mask shape:
        # for inp self_attention: [batch_size, 1, 1, key_len(inp)]
        # for tar self_attention: [batch_size, 1, query_len(tar)(=key_len(tar)), key_len(tar)(=query_len(tar))]
        # for encd_attention: [batch_size, 1, 1, key_len(inp)]
        if mask is not None:
            mask = mask.to(self.device)
            energy = energy.masked_fill(mask==0, -1e10)
        attention = torch.softmax(energy, axis=-1)
        print(f'query: {query.shape}\n{query[0][0]}\n\nkey: {key.shape}\n{key[0][0]}\n\nenergy: {energy.shape}\n{energy[0][0]}\n\nattention: {attention.shape}\n{attention[0][0]}\n\n')

        attention = self.dropout(attention)
        x = torch.matmul(attention, value)  # [batch_size, n_heads, query_len, head_dim]
        return x, attention, energy

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        x, attention, energy = self.scaled_dot_product_attention(query, key, value, mask)
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_len, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hidden_dim)  # [batch_size, query_len, hidden_dim]

        outputs = self.fc_o(x)
        
        return outputs, attention, energy


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, pf_dim, hidden_dim, dropout_ratio):
        super().__init__()
        self.fc_0 = nn.Linear(hidden_dim, pf_dim)
        self.fc_1 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs):
        x = torch.relu(self.fc_0(inputs))
        x = self.dropout(x)
        outputs = self.fc_1(x)
        return outputs


class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio

        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs, mask=None):
        attn_outputs, enc_attention, enc_energy = self.self_attention(inputs, inputs, inputs, mask)
        attn_outputs = self.dropout(attn_outputs)
        attn_outputs = self.self_attn_norm(inputs+attn_outputs)  # residual connection

        ff_outputs = self.pos_feedforward(attn_outputs)
        ff_outputs = self.dropout(ff_outputs)
        ff_outputs = self.pos_ff_norm(attn_outputs+ff_outputs)  # residual connection

        return ff_outputs, enc_attention, enc_energy  # [batch_size, query_len(inp), hidden_dim]


class DecoderLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.encd_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio, device)
        self.encd_attn_norm = nn.LayerNorm(hidden_dim)
        self.pos_feedforward = PositionwiseFeedforwardLayer(pf_dim, hidden_dim, dropout_ratio)
        self.pos_ff_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        self_attn_outputs, dec_attention, dec_energy = self.self_attention(target, target, target, target_mask)
        self_attn_outputs = self.dropout(self_attn_outputs)
        self_attn_outputs = self.self_attn_norm(target+self_attn_outputs)
        
        # self_attn_outputs shape: [batch_size, query_len(tar), hidden_dim]
        # encd shape: [batch_size, query_len(inp), hidden_dim]
        # new_query_len = query_len(tar); new_key_len(=new_val_len) = query_len(inp)
        encd_attn_outputs, attention, enc_dec_energy = self.encd_attention(self_attn_outputs, encd, encd, encd_mask)
        encd_attn_outputs = self.dropout(encd_attn_outputs)
        encd_attn_outputs = self.encd_attn_norm(self_attn_outputs+encd_attn_outputs)

        outputs = self.pos_feedforward(encd_attn_outputs)
        outputs = self.dropout(outputs)
        outputs = self.pos_ff_norm(encd_attn_outputs+outputs)

        return outputs, dec_attention, attention, dec_energy, enc_dec_energy  # [batch_size, query_len(tar), hidden_dim]


class EncoderPrenet(nn.Module):

    def __init__(self, n_layers, emb_size, hidden_dim, kernel_size=5, dropout_ratio=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([
                nn.Conv1d(
                    emb_size, hidden_dim,
                    kernel_size=kernel_size,
                    padding=int(kernel_size/2)
                )
        ] + [
                nn.Conv1d(
                    hidden_dim, hidden_dim,
                    kernel_size=kernel_size,
                    padding=int(kernel_size/2)
                )
                for _ in range(n_layers-1)
        ])
        self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout_ratio)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_embs):
        token_embs = token_embs.permute(0, 2, 1)
        for conv, batch_norm in zip(self.conv_layers, self.batch_norm_layers):
            token_embs = conv(token_embs)
            token_embs = batch_norm(token_embs)
            token_embs = torch.relu(token_embs)
            token_embs = self.dropout(token_embs)
        token_embs = token_embs.permute(0, 2, 1)
        outputs = self.projection(token_embs)
        return outputs


class Encoder(nn.Module):

    def __init__(self, input_dim, max_seq_len, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device):
        # input_dim = encoder vocab_size
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.tok_emb = nn.Embedding(input_dim, hidden_dim)
        self.encoder_prenet = EncoderPrenet(3, hidden_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.encd_stk = nn.ModuleList([
            EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.tok_emb(x)
        x = self.encoder_prenet(x)

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        emb = x * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        enc_attention = []
        for layer in self.encd_stk:
            outputs, enc_attention_tmp, enc_energy = layer(outputs, mask)
            enc_attention.append(enc_attention_tmp)

        return outputs, enc_attention, enc_energy


class DecoderPrenet(nn.Module):
    
    def __init__(self, n_mels, hidden_dim, dropout_ratio=0.2):
        super().__init__()
        self.fc_0 = nn.Linear(n_mels, hidden_dim//2)
        self.fc_1 = nn.Linear(hidden_dim//2, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, inputs):
        x = torch.relu(self.fc_0(inputs))
        x = self.dropout(x)
        x = torch.relu(self.fc_1(x))
        outputs = self.dropout(x)
        return outputs


class MelDecoder(nn.Module):
    
    def __init__(self, n_mels, max_seq_len, hidden_dim, n_layers, n_heads, pf_dim,
                 dropout_ratio, device):
        super().__init__()
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.decoder_prenet = DecoderPrenet(n_mels, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.decd_stk = nn.ModuleList([
            DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, device)
            for _ in range(n_layers)
        ])

        self.mel_linear = nn.Linear(hidden_dim, n_mels)
        self.stop_linear = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, target, encd, target_mask, encd_mask):
        batch_size = target.shape[0]
        seq_len = target.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        emb = self.decoder_prenet(target) * self.scale + self.pos_emb(pos)
        outputs = self.dropout(emb)

        dec_attention = []
        attention = []
        for layer in self.decd_stk:
            outputs, dec_attention_tmp, attention_tmp, dec_energy, enc_dec_energy = layer(outputs, encd, target_mask, encd_mask)
            dec_attention.append(dec_attention_tmp)
            attention.append(attention_tmp)

        mel_outputs = self.mel_linear(outputs)
        stop_prob = torch.sigmoid(self.stop_linear(outputs))

        return mel_outputs, stop_prob, dec_attention, attention, dec_energy, enc_dec_energy
        

class Transformer(nn.Module):

    def __init__(self, input_dim, n_mels, n_layers, hidden_dim, n_heads, pf_dim,
                 text_seq_len, speech_seq_len, pad_idx, dropout_ratio, device):
        super().__init__()
        self.device = device

        self.speech_seq_len = speech_seq_len

        self.encoder = Encoder(
            input_dim, text_seq_len, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device
        )
        self.decoder = MelDecoder(
            n_mels, speech_seq_len, hidden_dim, n_layers, n_heads, pf_dim,
            dropout_ratio, device
        )
        self.pad_idx = pad_idx

    def create_padding_mask(self, key, for_speech=False):
        if not for_speech:
            mask = key.ne(self.pad_idx).unsqueeze(1).unsqueeze(2)
        else:
            batch_size = key.shape[0]
            key_len = key.shape[1]
            mask = torch.max(key, axis=-1).values.ne(0)
            for_sos = torch.tensor([True]).repeat(batch_size, 1).to(self.device)
            mask = torch.concat([for_sos, mask[:, 1:]], axis=-1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            target_sub_mask = torch.tril(torch.ones((key_len, key_len))).bool().to(self.device)
            mask = mask & target_sub_mask
        return mask

    def forward(self, inp, tar):
        inp_mask = self.create_padding_mask(inp)
        tar_mask = self.create_padding_mask(tar, True)

        enc_inp, enc_attention, enc_energy = self.encoder(inp, inp_mask)
        mel_outputs, stop_prob, dec_attention, attention, dec_energy, enc_dec_energy = self.decoder(tar, enc_inp, tar_mask, inp_mask)

        return mel_outputs, stop_prob, enc_attention, dec_attention, attention, enc_energy, dec_energy, enc_dec_energy
