from typing import Optional
from torch import Tensor

from .base import Datastruct, dataclass, Transform


class IdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return IdentityDatastruct(transforms=self, **kwargs)

    def __repr__(self):
        return "IdentityTransform()"


@dataclass
class IdentityDatastruct(Datastruct):
    transforms: IdentityTransform

    features: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features"]

    def get_feats(self, feat_type = None):
        if feat_type == "rfeats":
            return self.features
        else:
            return self.features

    @property
    def joints(self):
        return self.features

    def __len__(self):
        return len(self.features)
