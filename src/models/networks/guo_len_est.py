from typing import List, Optional
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor
from src.utils.torch_utils import remove_padding
from torch.distributions.distribution import Distribution

class GuoLenEst(nn.Module):
    def __init__(self, textencoder, checkpoint_path, **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)

        if checkpoint_path != '':
            checkpoint = torch.load(checkpoint_path,
                                    map_location=self.textencoder.device)
            self.textencoder.load_state_dict(checkpoint['estimator'])
    
    def forward(self, batch, do_inference = False):
        ret = self.text_encoder_forward(
            batch["cap_lens"],
            batch["word_embs"],
            batch["pos_onehot"],
            )
        pred_dis = ret

        # GT data
        unit_length = 4
        gt_dis = [min(x // unit_length, (200 // unit_length)-1) for x in batch["orig_length"]]          # Add 200 in config file
        gt_dis = torch.tensor(gt_dis, device=pred_dis.device)

        # t1 = gt_dis.cpu().numpy()
        # t2 = torch.argmax(pred_dis, dim=1).detach().cpu().numpy()

        model_out = {
            'pred_dis': pred_dis,
            'gt_dis': gt_dis,
        }
        return model_out

    def text_encoder_forward(self, cap_lens, word_embs, pos_onehot):
        
        pred_dis = self.textencoder(word_embs, pos_onehot, cap_lens)

        return pred_dis
