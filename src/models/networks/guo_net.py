from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution

import random


class GuoNet(nn.Module):
    def __init__(self, textencoder, prior_network, posterior_network, decoder_network, 
                att_layer, transforms, vae, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.prior_network = instantiate(prior_network)
        self.posterior_network = instantiate(posterior_network, input_size = 1536)
        self.decoder_network = instantiate(decoder_network)
        # self.conv_encoder = instantiate(conv_encoder, nfeats = pose_dim)
        # self.conv_decoder = instantiate(conv_decoder, nfeats = pose_dim)
        self.att_layer = instantiate(att_layer)
        self.sample_mean = False
        self.fact = None
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

    def generate_sample(self, batch):
        datastruct_from_text = self.text_to_motion_forward(batch["text"],batch["length"])
        return remove_padding(datastruct_from_text.joints, batch["length"])
    
    def forward(self, batch, autoencoder, length_estimator, do_inference = False):
        ret = self.text_encoder_forward(
            batch["caption"],
            batch["cap_lens"],
            batch["word_embs"],
            batch["pos_onehot"],
            )
        word_hids, hidden = ret

        # Encode the motion/decode to a motion
        ret = self.motion_encode_forward(
            autoencoder,
            batch["datastruct"],
            batch["orig_length"],
            word_hids,
            hidden,
            batch["length"],
            return_latent = True
            )
        datastruct_from_text, gt_snippets, pred_snippets, mus_post, mus_pri, logvars_post, logvars_pri = ret

        # GT data
        datastruct_gt = batch["datastruct"]

        model_out = {
            'pred_data': datastruct_from_text,
            'gt_data': datastruct_gt,
            'gt_snippets': gt_snippets,
            'pred_snippets': pred_snippets,
            'mus_post': mus_post,
            'mus_pri': mus_pri,
            'logvars_post': logvars_post,
            'logvars_pri': logvars_pri
        }
        return model_out

    def motion_encode_forward(self, autoencoder, 
                                 datastruct,
                                 orig_lengths,
                                 word_hids,
                                 hidden,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):

        # Hyperparameter by Guo. Length of motion?
        unit_length = 4

        datastruct.transforms = self.transforms

        # Initialize first hidden state for vae gru models
        hidden_pos = self.posterior_network.get_init_hidden(hidden)
        hidden_pri = self.prior_network.get_init_hidden(hidden)
        hidden_dec = self.decoder_network.get_init_hidden(hidden)

        # Again last 4 frames were removed in Guo
        snippets = autoencoder.motionencoder(datastruct.features[..., :-4]).detach()
        # Initially input a mean vector
        # check for device
        mov_in = autoencoder.motionencoder(
            torch.zeros((datastruct.features.shape[0], unit_length, datastruct.features.shape[-1] - 4), device=snippets.device)     # Removed -4 from datastruct.features
        ).squeeze(1).detach()
        # schedule_len = 20           # As per Guo this should have been 6 for kit and 10 for t2m
        # assert snippets.shape[1] == schedule_len

        mus_pri = []
        logvars_pri = []
        mus_post = []
        logvars_post = []
        fake_mov_batch = []

        query_input = []

        for i in range(snippets.shape[1]):
            mov_tgt = snippets[:, i]

            # Feed input in attention layer
            att_vec, _ = self.att_layer(hidden_dec[-1], word_hids)
            query_input.append(hidden_dec[-1])

            # Check what is this m_len. Guo is getting from the batch. len(datastruct) is len(rfeats) ########
            tta = [(x // unit_length) - i for x in orig_lengths]

            # There is code for transformer but no mention in paper
            # if self.opt.text_enc_mod == 'bigru':
            pos_in = torch.cat([mov_in, mov_tgt, att_vec], dim=-1)
            pri_in = torch.cat([mov_in, att_vec], dim=-1)

            '''Posterior'''
            z_pos, mu_pos, logvar_pos, hidden_pos = self.posterior_network(pos_in, hidden_pos, tta)

            '''Prior'''
            z_pri, mu_pri, logvar_pri, hidden_pri = self.prior_network(pri_in, hidden_pri, tta)

            # See how to check if train or eval mode
            eval_mode = False

            '''Decoder'''
            if eval_mode:
                dec_in = torch.cat([mov_in, att_vec, z_pri], dim=-1)
            else:
                dec_in = torch.cat([mov_in, att_vec, z_pos], dim=-1)
            fake_mov, hidden_dec = self.decoder_network(dec_in, hidden_dec, tta)

            mus_post.append(mu_pos)
            logvars_post.append(logvar_pos)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)
            fake_mov_batch.append(fake_mov.unsqueeze(1))

            # Add parameter for teacher forcing
            tf_ratio = 0.4
            teacher_force = True if random.random() < tf_ratio else False

            if teacher_force:
                mov_in = snippets[:, i].detach()
            else:
                mov_in = fake_mov.detach()

        # Decode the latent vector to a motion
        fake_movements = torch.cat(fake_mov_batch, dim=1)

        # print(self.fake_movements.shape)

        # reco_data = self.fake_motions
        fake_motions = autoencoder.motiondecoder(fake_movements)
        # gt_data = datastruct.features

        mus_post = torch.cat(mus_post, dim=0)
        mus_pri = torch.cat(mus_pri, dim=0)
        logvars_post = torch.cat(logvars_post, dim=0)
        logvars_pri = torch.cat(logvars_pri, dim=0)

        # output = torch.cat([self.fake_motions, reco_data, gt_data], dim=0)
        # datastruct = self.Datastruct(features=output)
        datastruct = self.Datastruct(features=fake_motions)

        if not return_latent:
            return datastruct
        return datastruct, snippets, fake_movements, mus_post, logvars_post, mus_pri, logvars_pri

    def text_encoder_forward(self, captions, cap_lens, word_embs, pos_onehot):
        
        word_hids, hidden = self.textencoder(word_embs, pos_onehot, cap_lens)

        return word_hids, hidden

    def get_motion_embeddings(self, autoencoder, features):
        snippets = autoencoder.motionencoder(features).detach()
        return snippets

    def get_text_embeddings(self, features):
        pass
