from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution

class TEMOSNet(nn.Module):
    def __init__(self, textencoder, motionencoder, motiondecoder, transforms, pose_dim, vae, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder, nfeats = pose_dim)
        self.motiondecoder = instantiate(motiondecoder, nfeats = pose_dim)
        self.vae = vae
        self.sample_mean = False
        self.fact = None
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

    def generate_sample(self, batch):
        datastruct_from_text = self.text_to_motion_forward(batch["text"],batch["length"])
        return remove_padding(datastruct_from_text.joints, batch["length"])

    def forward(self, batch):
        ret = self.text_to_motion_forward(
            batch["text"],
            batch["length"],
            return_latent = True
            )
        datastruct_from_text, latent_from_text, distribution_from_text = ret

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(
            batch["datastruct"],
            batch["length"],
            return_latent = True
            )
        datastruct_from_motion, latent_from_motion, distribution_from_motion = ret

        # GT data
        datastruct_gt = batch["datastruct"]
        if self.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_gt = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_gt = None


        model_out = {
            'pred_data': datastruct_from_text,
            'pred_m2m': datastruct_from_motion,
            'gt_data': datastruct_gt,
            'dist_text': distribution_from_text,
            'dist_motion': distribution_from_motion,
            'dist_gt': distribution_gt,
            'latent_text': latent_from_text,
            'latent_motion': latent_from_motion
        }
        return model_out

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        if self.vae:
            distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution
