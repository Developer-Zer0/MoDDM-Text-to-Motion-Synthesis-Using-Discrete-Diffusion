import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor

import math

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, max_len: int = 300, **kwargs) -> None:

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, pos: Tensor) -> Tensor:
        return self.pe[pos]


class GRUGenerator(pl.LightningModule):
    def __init__(self,
                text_size: int = 1024, 
                input_size: int = 1152, 
                output_size: int = 512, 
                hidden_size: int = 1024, 
                n_layers: int = 1, **kwargs) -> None:

        super(GRUGenerator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.contact_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs: Tensor, hidden: list, pos) -> tuple[Tensor, list]:
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(pos).to(inputs.device).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        # pose_pred = self.output(h_in) + last_pred.detach()
        # contact = self.contact_net(pose_pred)
        # return torch.cat([pose_pred, contact], dim=-1), hidden
        return pose_pred, hidden

    # def forward(self, inputs: Tensor, last_pred: Tensor, hidden: list, pos):
    #     h_in = self.emb(inputs)
    #     pos_enc = self.positional_encoder(pos).to(inputs.device).detach()
    #     h_in = h_in + pos_enc
    #     for i in range(self.n_layers):
    #         # print(h_in.shape)
    #         hidden[i] = self.gru[i](h_in, hidden[i])
    #         h_in = hidden[i]
    #     pose_pred = self.output(h_in)
    #     # pose_pred = self.output(h_in) + last_pred.detach()
    #     # contact = self.contact_net(pose_pred)
    #     # return torch.cat([pose_pred, contact], dim=-1), hidden
    #     return pose_pred, hidden