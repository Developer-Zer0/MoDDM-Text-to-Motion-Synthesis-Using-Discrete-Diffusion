import random

import numpy as np
import torch
import torch.nn as nn
# from einops import rearrange
from hydra.utils import instantiate

# from src.models.motionencoder.resnet1d import Encoder, Decoder
from src.models.vqvae.modules.jukebox_bottleneck import BottleneckBlock
# from src.models.gesture.sq_vae_net import SQEmbedding
from src.models.networks.rq_vae_net import RQBottleneck
# from vector_quantize_pytorch import VectorQuantize as VQ_Enhance
# from src.models.modules.sq_quantizer import GaussianVectorQuantizer


class VQModel(nn.Module):
    def __init__(self,encoder, decoder, autoencoder_model_args, autoencoder_z_channels, transforms, pose_dim, **kwargs):
        super().__init__()
        self.transforms = instantiate(transforms)
        remap = None
        sane_index_shape = False # tell vector quantizer to return indices as bhw
        n_embed = autoencoder_model_args.n_embed
        embed_dim = autoencoder_model_args.embed_dim
        self.encoder = instantiate(encoder, nfeats = pose_dim)
        self.decoder = instantiate(decoder, nfeats = pose_dim)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)

        # self.bottleneck_quantize = BottleneckBlock(n_embed, embed_dim, 0.99)
        quantizer_type = autoencoder_model_args.quantizer_type
        self.quantizer_type = autoencoder_model_args.quantizer_type
        if quantizer_type == 'vq':
            self.bottleneck_quantize = BottleneckBlock(n_embed, embed_dim, 0.99)
        elif quantizer_type == 'sq':
            self.bottleneck_quantize = SQEmbedding("gaussian_1", n_embed, embed_dim)
        elif quantizer_type == "rq":
            self.bottleneck_quantize = RQBottleneck(n_embed, embed_dim, autoencoder_model_args.rq_residual_layers)
        else:
            self.bottleneck_quantize = BottleneckBlock(n_embed, embed_dim, 0.99)


        # self.vq_enhance = VQ_Enhance(
        #     dim = embed_dim,
        #     codebook_size = n_embed,
        #     codebook_dim = 32,
        #     use_cosine_sim = True,
        #     channel_last = False,
        #     kmeans_init = True,
        #     threshold_ema_dead_code = 2
        # )

        # self.vq_enhance = VQ_Enhance(
        #     dim = embed_dim,
        #     codebook_size = n_embed,
        #     #codebook_dim = 32,
        #     use_cosine_sim = False,
        #     channel_last = False,
        #     #kmeans_init = True,
        #     threshold_ema_dead_code = 2
        # )

        #self.sq_quantizer = GaussianVectorQuantizer(n_embed, embed_dim)


        self.quant_conv = torch.nn.Conv1d(autoencoder_z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, autoencoder_z_channels, 1)


    def run_vq(self, z):
        #quant, vq_loss, info = self.quantize(z)

        if self.quantizer_type == 'vq':
            # run jukebox vq
            _, quant, vq_loss, metrics = self.bottleneck_quantize(z)
        elif self.quantizer_type == 'sq':
            # run sq
            quant, vq_loss, perplexity = self.bottleneck_quantize(z, 1.0)
            metrics = {'perplexity': perplexity}
        elif self.quantizer_type == 'rq':
            # run rq
            quant, vq_loss, perplexity = self.bottleneck_quantize(torch.transpose(z, 1, 2))
            quant = torch.transpose(quant, 1, 2)
            metrics = {'perplexity': perplexity}
        #metrics = {'perplexity': perplexity}
        #quant, _, vq_loss = self.vq_enhance(z)
        return quant, vq_loss, metrics

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, metrics = self.run_vq(h)
        return quant, emb_loss, metrics

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

        # get only the quantize feature vector, used for training the autoregressive transformer
    @torch.no_grad()
    def encode_feature(self, x):
        # x = torch.flatten(x, -2, -1)
        x = torch.transpose(x, 1, 2)
        h = self.encoder(x)
        h = self.quant_conv(h)
        q = self.bottleneck_quantize.encode(h)
        return q

    @torch.no_grad()
    def encode_quantize(self, x):
        # x = torch.flatten(x, -2, -1)
        x = torch.transpose(x, 1, 2)
        h = self.encoder(x)
        h = self.quant_conv(h)
        q = self.bottleneck_quantize.quantise(h)
        return q

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.bottleneck_quantize, 'get_soft_codes')
        #xs = torch.transpose(xs, 1, 2)
        code = self.encode_feature(xs)
        z_e = self.bottleneck_quantize.embed_code(code)
        soft_code, code = self.bottleneck_quantize.get_soft_codes(z_e, temp = temp, stochastic = stochastic)
        return soft_code, code

    @torch.no_grad()
    def embed_code_with_depth(self, code):
        return self.bottleneck_quantize.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_seq(self, seq):
        x_d = self.bottleneck_quantize.decode(seq)
        reconstructed_x = self.decode(x_d)
        return reconstructed_x

    @torch.no_grad()
    def decode_rq_seq(self, seq):
        z = self.bottleneck_quantize.embed_code_with_depth(seq)

        # Change required to get same dimensions as in only vq-vae training 
        z = z.squeeze(2)

        z = torch.transpose(z, 1, 2)
        # print(z.shape)
        reconstructed_x = self.decode(z)
        return reconstructed_x

    def forward(self, x, do_inference = False):
        # poses = x['poses']['target_pose']
        datastruct = x['datastruct']
        poses = datastruct.features.float()

        # (B,T,J,C) -> (B, T, J*C)
        #poses = torch.flatten(poses, -2, -1)
        # (B, T, J*C) -> (B, J*C, T) for 1D Conv
        poses = torch.transpose(poses, 1, 2)
        quant, diff, metrics = self.encode(poses)
        dec = self.decode(quant)

        # convert back to (B, T, J*C)
        dec = torch.transpose(dec, 1, 2).double()
        datastruct_pred = self.transforms.Datastruct(features=dec)

        model_out = {
            'pred_data': datastruct_pred,
            'codebook_loss': diff, # codebook loss is already computed
            'gt_data': datastruct,
        }
        #model_out.update(metrics)
        
        return model_out



