import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class AttLayer(pl.LightningModule):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
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

class MotionEarlyAttDecoder(pl.LightningModule):
    def __init__(self, n_mot_vocab, init_hidden_size, hidden_size, n_layers, **kwargs):
        super(MotionEarlyAttDecoder, self).__init__()

        self.input_emb = nn.Embedding(n_mot_vocab, hidden_size)
        self.n_layers = n_layers

        self.z2init = nn.Linear(init_hidden_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.att_layer = AttLayer(hidden_size, init_hidden_size, hidden_size)
        self.att_linear = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.trg_word_prj = nn.Linear(hidden_size, n_mot_vocab, bias=False)

        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.att_layer.apply(init_weight)
        self.att_linear.apply(init_weight)
        self.hidden_size = hidden_size

        self.trg_word_prj.weight = self.input_emb.weight

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    # input(batch_size, dim)
    def forward(self, src_output, inputs, hidden):
        h_in = self.input_emb(inputs)

        # h_in *= self.hidden_size ** 0.5

        att_vec, _ = self.att_layer(hidden[-1], src_output)
        # print(att_vec.shape, h_in.shape)
        h_in = self.att_linear(
            torch.cat([att_vec, h_in], dim=-1)
        )

        for i in range(self.n_layers):
            # print(h_in.shape, hidden[i].shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        # print(h_in.shape, src_output.shape)
        pred_probs = self.trg_word_prj(h_in)
        return pred_probs, hidden
