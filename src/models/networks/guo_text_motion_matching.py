from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution

import numpy as np


class GuoTextMotionMatching(nn.Module):
    def __init__(self, textencoder, motion_encoder, transforms, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.motion_encoder = instantiate(motion_encoder)
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

    def generate_sample(self, batch):
        datastruct_from_text = self.text_to_motion_forward(batch["text"], batch["length"])
        return remove_padding(datastruct_from_text.joints, batch["length"])

    def forward(self, batch, autoencoder, length_estimator, do_inference=False):
        ret = self.text_encoder_forward(
            batch["caption"],
            batch["cap_lens"],
            batch["word_embs"],
            batch["pos_onehot"],
        )
        text_embedding = ret

        # Encode the motion/decode to a motion
        ret = self.motion_encode_forward(
            autoencoder,
            batch["datastruct"],
            batch["cap_lens"],
        )
        motion_embedding = ret

        # GT data
        datastruct_gt = batch["datastruct"]

        model_out = {
            'pred_data': motion_embedding,
            'gt_data': text_embedding,
        }
        return model_out

    def motion_encode_forward(self, autoencoder,
                              datastruct,
                              orig_lengths,
                              ):

        # Hyperparameter by Guo. Length of motion?
        unit_length = 4

        datastruct.transforms = self.transforms

        movements = autoencoder.motionencoder(datastruct.features[..., :-4]).detach()
        orig_lengths = [(x // unit_length) for x in orig_lengths]
        motion_embedding = self.motion_encoder(movements, orig_lengths)

        return motion_embedding

    def text_encoder_forward(self, captions, cap_lens, word_embs, pos_onehot):

        hidden = self.textencoder(word_embs, pos_onehot, cap_lens)

        return hidden

    def get_motion_embeddings(self, autoencoder, features, m_lens):

        unit_length = 4

        align_idx = np.argsort(m_lens)[::-1].copy()
        features = features[align_idx]
        m_lens = np.array(m_lens)[align_idx]

        movements = autoencoder.motionencoder(features[..., :-4]).detach()
        orig_lengths = [(x // unit_length) for x in m_lens]
        motion_embedding = self.motion_encoder(movements, orig_lengths)

        return motion_embedding

    def get_text_embeddings(self, word_embs, pos_ohot, cap_lens, m_lens):
        align_idx = np.argsort(np.array(m_lens))[::-1].copy()
        text_embedding = self.textencoder(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        return text_embedding
