import os
import sys

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

# import temos.launch.blender
# import temos.launch.prepare  # noqa
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# @hydra.main(version_base=None, config_path="/home/ICT2000/achemburkar/Desktop/TextMotionGenerator/src/utils/temos_render/", config_name="render")
def _render_cli(filename):
    print('#################hey1')
    return render_cli(filename)


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for index in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths



def render_cli(filename) -> None:
    cfg = {
        'npy': filename,
        'folder': None,
        'mode': 'video',
        'vid_ext': 'webm',
        'denoising': True,
        'oldrender': True,
        'res': 'high',
        'exact_frame': 0.1,
        'faces_path': '/data/feng/Gestures/Deps/TextMotionDeps/smplh/smplh.faces',
        'downsample': False,
        'gt': False,
        'num': 8,
        'always_on_floor': True,
        'canonicalize': True,
    }
    if cfg['npy'] is None:
        if cfg['folder'] is None or cfg.split is None:
            raise ValueError("You should either use npy=XXX.npy, or folder=XXX and split=XXX")
        # # only them can be rendered for now
        # if not cfg.infolder:
        #     jointstype = cfg.jointstype
        #     assert ("mmm" in jointstype) or jointstype == "vertices"
        #
        # from temos.data.utils import get_split_keyids
        # from pathlib import Path
        # from evaluate import get_samples_folder
        # from sample import cfg_mean_nsamples_resolution, get_path
        # keyids = get_split_keyids(path=Path(cfg.path.datasets)/ "kit-splits", split=cfg.split)
        #
        # onesample = cfg_mean_nsamples_resolution(cfg)
        # if not cfg.infolder:
        #     model_samples, amass, jointstype = get_samples_folder(cfg.folder,
        #                                                           jointstype=cfg.jointstype)
        #     path = get_path(model_samples, amass, cfg.gender, cfg.split, onesample, cfg.mean, cfg.fact)
        # else:
        #     path = Path(cfg.folder)
        #
        # paths = extend_paths(path, keyids, onesample=onesample, number_of_samples=cfg.number_of_samples)
    else:
        paths = [cfg['npy']]

    from blender import render
    from video import Video
    import numpy as np

    init = True
    for path in paths:
        print('#################hey2')
        try:
            data = np.load(path)
            data = np.transpose(data, (2, 0, 1))[..., [2, 0, 1]]
            data[..., :, :, 0] = -1*data[..., :, :, 0]
            data[..., :, :, 1] = -1*data[..., :, :, 1]
        except FileNotFoundError:
            logger.info(f"{path} not found")
            print('#################hey3', path)
            continue

        if cfg['mode'] == "video":
            frames_folder = path.replace(".npy", "_frames")
        else:
            frames_folder = path.replace(".npy", ".png")

        if cfg['mode'] == "video":
            vid_path = path.replace(".npy", f".{cfg['vid_ext']}")
            if os.path.exists(vid_path):
                continue

        out = render(data, frames_folder,
                     denoising=cfg['denoising'],
                     oldrender=cfg['oldrender'],
                     res=cfg['res'],
                     canonicalize=cfg['canonicalize'],
                     exact_frame=cfg['exact_frame'],
                     num=cfg['num'], mode=cfg['mode'],
                     faces_path=cfg['faces_path'],
                     downsample=cfg['downsample'],
                     always_on_floor=cfg['always_on_floor'],
                     init=init,
                     gt=cfg['gt'],
                     )

        init = False

        if cfg['mode'] == "video":
            if cfg['downsample']:
                video = Video(frames_folder, fps=12.5, res=cfg['res'])
            else:
                video = Video(frames_folder, fps=15, res=cfg['res'])

            video.save(out_path=vid_path)
            logger.info(vid_path)

        else:
            logger.info(f"Frame generated at: {out}")


if __name__ == '__main__':
    folder_path = '/home/ICT2000/achemburkar/Desktop/TextMotionGenerator/temp/'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith("vertices") and file.endswith(".npy"):
                filename = os.path.join(root, file)
                _render_cli(filename)