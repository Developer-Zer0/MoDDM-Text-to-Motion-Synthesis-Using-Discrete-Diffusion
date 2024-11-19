from typing import Optional
from torch import Tensor

from .base import Datastruct, dataclass, Transform

from .rots2rfeats import Rots2Rfeats
from .rots2joints import Rots2Joints
from .joints2jfeats import Joints2Jfeats

from scipy.ndimage import gaussian_filter

import torch

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_root_rot_pos(data):
    data = data.float()
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4].float()
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape) 
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

class HumanMLSMPLTransform(Transform):
    def __init__(self, rots2rfeats: Rots2Rfeats,
                 rots2joints: Rots2Joints,
                 joints2jfeats: Joints2Jfeats,
                 joints_num: int = 22,
                 **kwargs):
        self.rots2rfeats = rots2rfeats
        self.rots2joints = rots2joints
        self.joints2jfeats = joints2jfeats
        self.joints_num = joints_num

    def Datastruct(self, **kwargs):
        return HumanMLSMPLDatastruct(_rots2rfeats=self.rots2rfeats,
                              _rots2joints=self.rots2joints,
                              _joints2jfeats=self.joints2jfeats,
                              transforms=self,
                              joints_num=self.joints_num,
                              **kwargs)

    def __repr__(self):
        return "SMPLTransform()"


class HumanMLRotIdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return HumanMLRotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"


@dataclass
class HumanMLRotTransDatastruct(Datastruct):
    rots: Tensor
    trans: Tensor

    transforms: HumanMLRotIdentityTransform = HumanMLRotIdentityTransform()

    def __post_init__(self):
        self.datakeys = ["rots", "trans"]

    def __len__(self):
        return len(self.rots)


@dataclass
class HumanMLSMPLDatastruct(Datastruct):
    transforms: HumanMLSMPLTransform
    _rots2rfeats: Rots2Rfeats
    _rots2joints: Rots2Joints
    _joints2jfeats: Joints2Jfeats
    joints_num: int = 22

    features: Optional[Tensor] = None
    rots_: Optional[HumanMLRotTransDatastruct] = None
    rfeats_: Optional[Tensor] = None
    joints_: Optional[Tensor] = None
    jfeats_: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features", "rots_", "rfeats_",
                         "joints_", "jfeats_"]
        # starting point
        if self.features is not None and self.rfeats_ is None:
            # t = torch.cat([self.features[..., 3], self.features[..., 1:3], self.features[..., 136:268]])
            self.rfeats_ = self.features

    # @property
    # def rots(self):
    #     # Cached value
    #     if self.rots_ is not None:
    #         return self.rots_
    #
    #     # self.rfeats_ should be defined
    #     assert self.rfeats_ is not None
    #
    #     self._rots2rfeats.to(self.rfeats.device)
    #     self.rots_ = self._rots2rfeats.inverse(self.rfeats)
    #     return self.rots_
    #
    # @property
    # def rfeats(self):
    #     # Cached value
    #     if self.rfeats_ is not None:
    #         return self.rfeats_
    #
    #     # self.rots_ should be defined
    #     assert self.rots_ is not None
    #
    #     self._rots2rfeats.to(self.rots.device)
    #     self.rfeats_ = self._rots2rfeats(self.rots)
    #     return self.rfeats_

    @property
    def joints(self):
        # Cached value
        if self.joints_ is not None:
            return self.joints_

        # self._rots2joints.to(self.rots.device)

        # joints_num = 22

        self.joints_ = recover_from_ric(self.features, self.joints_num)
        # self.joints_ = motion_temporal_filter(self.joints_.cpu().numpy())
        # self.joints_ = self._rots2joints(self.rots)
        return self.joints_

    # @property
    # def jfeats(self):
    #     # Cached value
    #     if self.jfeats_ is not None:
    #         return self.jfeats_
    #
    #     self._joints2jfeats.to(self.joints.device)
    #     self.jfeats_ = self._joints2jfeats(self.joints)
    #     return self.jfeats_

    # features for computing the loss function
    def get_feats(self, feat_type = None):
        if feat_type == "rfeats":
            return self.features
        else:
            return self.features

    def __len__(self):
        return len(self.features)
