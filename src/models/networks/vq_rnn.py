from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
import random

import matplotlib.pyplot as plt


class VQRNN(nn.Module):
    def __init__(self, textencoder, motionencoder, transforms, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct
        # self.device = next(self.parameters()).device

    def forward(self, batch, autoencoder, length_estimator, do_inference=False):

        # Shuhong's vq-vae
        # motion = batch['datastruct'].features.float()
        # quant = autoencoder.encode_feature(motion)
        # quant = quant.squeeze(2)

        # Guo's vq-vae
        motion = batch['datastruct'].features
        motion = motion.detach().to(autoencoder.motionencoder.device).float()
        pre_latents = autoencoder.motionencoder(motion[..., :-4])
        _, _, quant, _ = autoencoder.quantizer(pre_latents)
        quant = quant.view(motion.shape[0], -1)

        # Hardcoded sos index and eos index as codebook size+1 and codebook size+2
        sos_tensor = 1024 * torch.ones((motion.shape[0], 1), device=quant.device).long()
        # eos_tensor = 1025*torch.ones((motion.shape[0], 1), device=quant.device).long()
        trg_input = torch.cat((sos_tensor, quant), 1)

        txt_hid_seq, txt_hid_last = self.textencoder(batch['word_embs'], batch['pos_onehot'], batch['cap_lens'])
        motion_hidden = self.motionencoder.get_init_hidden(txt_hid_last)

        flipped_coin = random.random()

        # TODO change
        tf_ratio = 0.5

        is_tf = True if flipped_coin < tf_ratio else False

        trg_len = trg_input.shape[1]
        trg_step_input = trg_input[:, 0]
        output_probs = []
        output_tokens = []
        for i in range(0, trg_len):
            trg_step_pred, text_hidden = self.motionencoder(txt_hid_seq, trg_step_input.detach(), motion_hidden)
            output_probs.append(trg_step_pred.unsqueeze(1))
            _, next_inp = trg_step_pred[:, 1:-1].max(-1)
            output_tokens.append(next_inp)
            if is_tf and i + 1 < trg_len:
                trg_step_input = trg_input[:, i + 1]
            else:
                trg_step_input = next_inp

        output_probs = torch.cat(output_probs, dim=1)[:, :-1, :]
        # output_tokens = torch.cat(output_tokens, dim=1)
        output_tokens = torch.stack(output_tokens, dim=1)[:, :-1]

        # Check this
        single_step_out = output_tokens.view(-1, output_tokens.shape[-1]).clone()

        # Shuhong's vq-vae
        # single_step_out = single_step_out.unsqueeze(2)
        # single_step_out = autoencoder.decode_rq_seq(single_step_out)
        # single_step_out = torch.transpose(single_step_out, 1, 2)

        # Guo's vq-vae
        single_step_out = autoencoder.quantizer.get_codebook_entry(single_step_out)
        single_step_out = autoencoder.motiondecoder(single_step_out)

        single_step_out = self.transforms.Datastruct(features=single_step_out)

        if do_inference:
            inference_output_tokens = []
            max_steps = 49
            trg_step_input = 1024 * torch.ones((motion.shape[0], 1), device=quant.device).long()
            trg_step_input = trg_step_input[:, 0]
            for i in range(max_steps+1):
                trg_step_pred, text_hidden = self.motionencoder(txt_hid_seq, trg_step_input.detach(),
                                                                motion_hidden)
                _, trg_step_input = trg_step_pred[:, 1:-1].max(-1)
                inference_output_tokens.append(trg_step_input)

            inference_output_tokens = torch.stack(inference_output_tokens, dim=1)[:, :-1]
            inference_out = inference_output_tokens.view(-1, inference_output_tokens.shape[-1]).clone()

            # Shuhong's vq-vae
            # inference_out = inference_out.unsqueeze(2)
            # inference_out = autoencoder.decode_rq_seq(inference_out)
            # inference_out = torch.transpose(inference_out, 1, 2).double()

            # Guo's vq-vae
            inference_out = autoencoder.quantizer.get_codebook_entry(inference_out)
            inference_out = autoencoder.motiondecoder(inference_out)

            inference_out = self.transforms.Datastruct(features=inference_out)

        # test = autoencoder.decode_rq_seq(quant.unsqueeze(2))
        # test = torch.transpose(test, 1, 2)
        # datastruct_test = self.transforms.Datastruct(features=test)

        # GT data
        datastruct_gt = batch['datastruct']

        if do_inference:
            model_out = {
                'pred_data': inference_out,
                # 'pred_data': single_step_out,
                'pred_single_step': single_step_out,
                'gt_data': datastruct_gt,
                'pred_tokens': output_probs.reshape(-1, output_probs.shape[-1]).clone(),
                'gt_tokens': quant.contiguous().view(-1).clone(),
                # 'datastruct_test': datastruct_test,
            }
        else:
            model_out = {
                'pred_data': single_step_out,
                'gt_data': datastruct_gt,
                'pred_tokens': output_probs.reshape(-1, output_probs.shape[-1]).clone(),
                'gt_tokens': quant.contiguous().view(-1).clone(),
                # 'datastruct_test': datastruct_test,
            }

        return model_out

    # def schedule_tf(self, epoch, start_tf=0.9, end_tf=0.1, end_epoch=35):
    #     tf_epoch = epoch if epoch < end_epoch else end_epoch
    #     return start_tf - (start_tf - end_tf) * tf_epoch / end_epoch

    def get_motion_embeddings(self, autoencoder, features):
        # quant = autoencoder.encode_feature(features)
        poses = torch.transpose(features, 1, 2)
        quant, diff, metrics = autoencoder.encode(poses)
        # datastruct_test = self.transforms.Datastruct(features=test)
        return quant

    def get_text_embeddings(self, features):
        text_emb = self.textencoder(features)
        text_emb = text_emb.unsqueeze(1)
        return text_emb
