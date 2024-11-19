from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution


class GuoAutoEncoder(nn.Module):
    def __init__(self, motionencoder, motiondecoder, transforms, pose_dim, vae: bool = False, **kwargs):

        super().__init__()
        self.motionencoder = instantiate(motionencoder, nfeats = pose_dim)
        self.motiondecoder = instantiate(motiondecoder, nfeats = pose_dim)
        self.vae = vae
        self.sample_mean = False
        self.fact = None
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

    def forward(self, batch, do_inference = False):

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(
            batch["datastruct"],
            batch["length"],
            return_snippet = True
            )
        datastruct_from_motion, snippets_from_motion = ret

        # GT data
        datastruct_gt = batch["datastruct"]
        # if self.vae:
        #     # Create a centred normal distribution to compare with
        #     mu_ref = torch.zeros_like(distribution_from_text.loc)
        #     scale_ref = torch.ones_like(distribution_from_text.scale)
        #     distribution_gt = torch.distributions.Normal(mu_ref, scale_ref)
        # else:
        #     distribution_gt = None


        model_out = {
            'pred_data': None,
            'pred_m2m': datastruct_from_motion,
            'gt_data': datastruct_gt,
            'snippets_motion': snippets_from_motion
        }
        return model_out

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_snippet: bool = True
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms

        # Encode the motion to the snippets
        snippets = self.motionencoder(datastruct.features)

        # Decode the latent vector to a motion
        features = self.motiondecoder(snippets)
        datastruct = self.Datastruct(features=features)

        if not return_snippet:
            return datastruct
        return datastruct, snippets
