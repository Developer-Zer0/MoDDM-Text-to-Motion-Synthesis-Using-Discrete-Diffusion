import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import sys
import numpy as np
# from os.path import join as pjoin
from src.utils.render_utils import render_animation
import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="sample_generation.yaml")
def main(cfg: DictConfig):

    with open("sample_description.txt", 'r') as t:
        text = t.readlines()

    mean = np.load('mean.npy')
    std = np.load('std.npy')

    model: LightningModule = hydra.utils.instantiate(cfg.model, nfeats=263, _recursive_=False)

    model.eval()

    generator = model.generator
    autoencoder = model.autoencoder

    generation_out = generator.generate(autoencoder, text)
    generation_out.features = generation_out.features.detach().cpu() * std + mean
    joints_np = generation_out.joints.cpu().numpy()[0]  # only one batch
    render_animation(joints_np, title=text[0], output="generation.mp4", dataset_name="HumanML3D")
    print("DONE!!")

if __name__ == "__main__":
    main()