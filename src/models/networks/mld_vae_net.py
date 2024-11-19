from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.models.utils.mld_model_utils.losses import MLDLosses

# class KLLoss:
#
#     def __init__(self):
#         pass
#
#     def __call__(self, q, p):
#         div = torch.distributions.kl_divergence(q, p)
#         return div.mean()
#
#     def __repr__(self):
#         return "KLLoss()"

class MLDVae(nn.Module):
    def __init__(self, loss_param_dict, vae_model, transforms, pose_dim, vae: bool = False, **kwargs):

        super().__init__()
        self.vae_model = instantiate(vae_model, nfeats = pose_dim)
        # self.motiondecoder = instantiate(motiondecoder, nfeats = pose_dim)
        self.vae = vae
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        # self.losses = MLDLosses(vae=self.vae, mode="xyz", cfg=loss_param_dict)

        self.loss_param_dict = loss_param_dict
        # self.recon_loss = torch.nn.SmoothL1Loss(reduction='mean')
        # self.kl_loss = KLLoss()

        self.vae_type = "mld"
        self.condition = "text"

    def forward(self, batch, do_inference = False):

        feats_ref = batch['datastruct'].features.float()
        # feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae_model.encode(feats_ref, lengths)
            feats_rst = self.vae_model.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae_model.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            # joints_rst = self.feats2joints(feats_rst)
            # joints_ref = self.feats2joints(feats_ref)
            joints_rst = self.transforms.Datastruct(features=feats_rst).joints
            joints_ref = self.transforms.Datastruct(features=feats_ref).joints
        # elif self.condition == "action":
        #     mask = batch["mask"]
        #     joints_rst = self.feats2joints(feats_rst, mask)
        #     joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }

        # loss = self.losses.update(rs_set)
        # if loss is None:
        #     raise ValueError(
        #         "Loss is None, this happend with torchmetrics > 0.7")
        # loss: float = 0.0
        # loss += self.recon_loss(rs_set['m_rst'], rs_set['m_ref']) * self.loss_param_dict['LAMBDA_REC']
        # loss += self.recon_loss(rs_set['joints_rst'], rs_set['joints_ref']) * self.loss_param_dict['LAMBDA_REC']
        # loss += self.kl_loss(rs_set['dist_m'], rs_set['dist_ref']) * self.loss_param_dict['LAMBDA_KL']

        pred_data = self.transforms.Datastruct(features=feats_rst)
        # datastruct_test = self.transforms.Datastruct(features=test)

        # n1 = feats_rst.cpu().numpy()
        # n2 = feats_ref.cpu().numpy()

        # GT data
        datastruct_gt = batch['datastruct']

        if True:
            model_out = {
                'pred_data': pred_data,
                'gt_data': datastruct_gt,
                # 'losses': loss,
                'rs_set': rs_set,
                # 'datastruct_test': datastruct_test,
            }

        return model_out
