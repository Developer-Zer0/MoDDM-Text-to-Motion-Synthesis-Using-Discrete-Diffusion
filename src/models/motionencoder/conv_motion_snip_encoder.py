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

class ConvMotionSnipEncoder(pl.LightningModule):
    ## What should be input size for encoder? Should we subtract 4?
    ## Is ff_size=hidden_size?

    def __init__(self, nfeats: int, latent_size: int = 512, hidden_size: int = 384, 
                dropout: float = 0.2, kernel_size: int = 4, 
                stride: int = 2, padding: int = 1, activation: str = "LeakyReLU", **kwargs) -> None:

        super(ConvMotionSnipEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(nfeats-4, hidden_size, kernel_size, stride, padding),
            nn.Dropout(dropout, inplace=True),

            ## Change LeakyReLU to be dynamic: refer https://discuss.pytorch.org/t/call-activation-function-from-string/30857/3
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_size, latent_size, kernel_size, stride, padding),
            nn.Dropout(dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(latent_size, latent_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)