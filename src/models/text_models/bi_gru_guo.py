import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TextEncoderBiGRU(pl.LightningModule):

    def __init__(self, 
                word_size: int = 300,                   # GloVe embedding size
                pos_size: int = 15,                     # One hot encoding size of POS embeddings (len(POS embedding))
                hidden_size: int = 512, **kwargs) -> None:
        
        super(TextEncoderBiGRU, self).__init__()
        # self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    
    # input(batch_size, seq_len, dim)
    def forward(self, 
            word_embs: Tensor,                          # Word embeddings from GloVe etc.
            pos_onehot: Tensor,                         # One-hot encodings for positional embeddings
            cap_lens                                    # Length of texts for pack_padded_sequence
            ) -> tuple[Tensor, Tensor]:
        
        word_embs = torch.tensor(word_embs, device=self.hidden.device).float()
        pos_onehot = torch.tensor(pos_onehot, device=self.hidden.device).float()
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        # cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)        # Enforce_sorted added, not in Guo

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(cap_lens):
            backward_seq[i:i+1, :length] = torch.flip(backward_seq[i:i+1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class AttLayer(pl.LightningModule):

    def __init__(self, 
                query_dim: int = 1024, 
                key_dim: int = 1024, 
                value_dim: int = 512, **kwargs) -> None:

        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, 
                query: Tensor, 
                key_mat: Tensor) -> tuple[Tensor, Tensor]:
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)       # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)                     # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)                     # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)              # (batch, seq_len, 1)
        values = val_set * co_weights                   # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)                        # (batch, value_dim)
        return pred, co_weights

    def short_cut(self, 
                querys: Tensor, 
                keys: Tensor) -> tuple[Tensor, Tensor]:
        return self.W_q(querys), self.W_k(keys)