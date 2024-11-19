from typing import List, Union
import torch.nn as nn
import os
import pytorch_lightning as pl
from torch import Tensor

class DistilBERTEncoder(pl.LightningModule):
    def __init__(self, modelpath: str = 'bert-base-uncased',
                 finetune: bool = False,
                 latent_dim: int = 32) -> None:
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size
        self.latent_dim = latent_dim
        # output feature dimenstion for each text token
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_encoded_dim, self.latent_dim)
            )

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False
                              ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            # Don't put the text_model in training
            if module == self.text_model:
                continue
            module.train(mode)
        return self

    # a simple encoder based on pre-trained DistillBertto get a fixed size feature vectors from a sentence
    def forward(self, texts: List[str]):
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask = True)
        x = self.projection(text_encoded)
        return x, mask
