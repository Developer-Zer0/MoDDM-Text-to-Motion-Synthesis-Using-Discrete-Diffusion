import numpy as np
import pandas
import torch
import logging
from torch import nn
from typing import Dict, Optional
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from pathlib import Path
import os

from src.datamodules.datasets.data_utils import get_split_keyids, smpl_data_to_matrix_and_trans, subsample
from src.datamodules.datasets.transforms import Transform

from src.datamodules.datasets.word_vectorizer import WordVectorizer, POS_enumerator
import codecs as cs
import spacy
import os
from os.path import join as pjoin

logger = logging.getLogger(__name__)

class KIT(Dataset):
    dataname = "KIT Motion-Language"
    def __init__(self, datapath: str,
                 splitpath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 correspondance_path: str = None,
                 amass_path: str = None,
                 smplh_path: str = None,
                 deps: str = None,
                 sampler=None,
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                 load_amass_data=False,
                 load_with_rot=False,
                 downsample=True,
                 tiny: bool = False, **kwargs):

        self.nlp = spacy.load('en_core_web_sm')
        vectorizer_path = deps + '/word_vectorizer'
        self.w_vectorizer = WordVectorizer(vectorizer_path, 'our_vab')
        self.split = split
        self.load_amass_data = load_amass_data
        self.load_with_rot = load_with_rot
        self.downsample = downsample

        if load_amass_data and not self.load_with_rot:
            self.transforms_xyz = transforms_xyz
            self.transforms_smpl = transforms_smpl
            self.transforms = transforms_xyz
        else:
            self.transforms = transforms

        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split)

        features_data = {}
        texts_data = {}
        durations = {}

        if load_amass_data:
            with open(correspondance_path) as correspondance_path_file:
                kitml_correspondances = json.load(correspondance_path_file)

        if progress_bar:
            enumerator = enumerate(tqdm(keyids, f"Loading KIT {split}"))
        else:
            enumerator = enumerate(keyids)

        if tiny:
            maxdata = 2
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        num_bad = 0
        if load_amass_data:
            bad_smpl = 0
            good_smpl = 0

        for i, keyid in enumerator:
            if len(features_data) >= maxdata:
                break

            anndata, success = load_annotation(keyid, datapath)
            if not success:
                logger.error(f"{keyid} has no annotations")
                continue

            # read smpl params
            if load_amass_data:
                smpl_data, success = load_amass_keyid(keyid, amass_path,
                                                      correspondances=kitml_correspondances)
                if not success:
                    bad_smpl += 1
                    continue
                else:
                    good_smpl += 1

                smpl_data, duration = downsample_amass(smpl_data, downsample=self.downsample, framerate=framerate)
                smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
            # read xyz joints in MMM format
            else:
                joints = load_mmm_keyid(keyid, datapath)
                joints, duration = downsample_mmm(joints, downsample=self.downsample, framerate=framerate)

            if split != "test" and not tiny:
                # Accept or not the sample, based on the duration
                if not self.sampler.accept(duration):
                    num_bad += 1
                    continue

            # Load rotation features (rfeats) datamodule from AMASS
            if load_amass_data and load_with_rot:
                features = self.transforms.rots2rfeats(smpl_data)
            # Load xyz features (jfeats) datamodule from AMASS
            elif load_amass_data and not load_with_rot:
                joints = self.transforms_smpl.rots2joints(smpl_data)
                features = self.transforms_xyz.joints2jfeats(joints)
            # Load xyz features (jfeats) datamodule from MMM
            else:
                features = self.transforms.joints2jfeats(joints)

            features_data[keyid] = features
            texts_data[keyid] = anndata
            durations[keyid] = duration

        if load_amass_data and not tiny:
            percentage = 100 * bad_smpl / (bad_smpl + good_smpl)
            logger.info(f"There are {bad_smpl} sequences not found ({percentage:.4}%) in AMASS.")

        if split != "test" and not tiny:
            total = len(features_data)
            percentage = 100 * num_bad / (total+num_bad)
            logger.info(f"There are {num_bad} sequences rejected by the sampler ({percentage:.4}%).")

        self.features_data = features_data
        self.texts_data = texts_data

        self.keyids = list(features_data.keys())
        self._split_index = list(self.keyids)
        self._num_frames_in_sequence = durations
        self.nfeats = len(self[0]["datastruct"].features[0])

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.features_data[keyid][frame_ix]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def sent_to_tokens(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list
    
    def guo_preprocess_text(self, text):

        word_list, pos_list = self.sent_to_tokens(text)
        tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
        # Pad with UNK will tokens length = max_text_length
        # max_text_len has been taken as 20 by Guo
        max_text_len = 20

        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, sent_len
    
    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]

        if True:
            word_embs, pos_onehot, sent_len = self.guo_preprocess_text(text)
            output_dict = {'word_embs': word_embs, 'pos_onehot': pos_onehot, 'caption': text, 'cap_lens': sent_len}
            return output_dict
        else:
            return text

    def load_keyid(self, keyid):
        num_frames = self._num_frames_in_sequence[keyid]
        frame_ix = self.sampler(num_frames)

        datastruct = self._load_datastruct(keyid, frame_ix)
        text = self._load_text(keyid)
        element = {"datastruct": datastruct, "text": text,
                   "length": len(datastruct), "keyid": keyid, "orig_length": num_frames}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"


def load_annotation(keyid, datapath):
    metapath = datapath / (keyid + "_meta.json")
    metadata = json.load(metapath.open())

    if metadata["nb_annotations"] == 0:
        logger.error(f"{keyid} has no annotations")
        return None, False

    annpath = datapath / (keyid + "_annotations.json")
    anndata = json.load(annpath.open())
    assert len(anndata) == metadata["nb_annotations"]
    return anndata, True


def load_mmm_keyid(keyid, datapath):
    xyzpath = datapath / (keyid + "_fke.csv")
    xyzdata = pandas.read_csv(xyzpath, index_col=0)
    joints = np.array(xyzdata).reshape(-1, 21, 3)
    return joints

def downsample_mmm(joints, *, downsample, framerate):
    nframes_total = len(joints)
    last_framerate = 100

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)
    joints = torch.from_numpy(joints[frames]).float()
    return joints, duration

def load_amass_keyid(keyid, amass_path, *, correspondances):
    identifier = correspondances[keyid]["identifier"]
    smpl_keyid_path = correspondances[keyid]["path"]

    if identifier == "kit":
        smpl_datapath = Path(amass_path) / "KIT" /  smpl_keyid_path
    elif identifier == "cmu":
        smpl_datapath = Path(amass_path) / "CMU" /  smpl_keyid_path

        if not os.path.exists(smpl_datapath):
            # try with EKUT folder instead
            smpl_datapath = Path(amass_path) / "EKUT" / smpl_keyid_path

            # File not found
            if not os.path.exists(smpl_datapath):
                return None, False
    else:
        raise TypeError(f"{identifier} identifier not recognized.")
    try:
        smpl_data = np.load(smpl_datapath)
    except FileNotFoundError:
        return None, False

    smpl_data = {x: smpl_data[x] for x in smpl_data.files}
    return smpl_data, True

def downsample_amass(smpl_data, *, downsample, framerate):
    nframes_total = len(smpl_data["poses"])
    last_framerate = smpl_data["mocap_framerate"].item()

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)

    # subsample
    smpl_data = {"poses": torch.from_numpy(smpl_data["poses"][frames]).float(),
                 "trans": torch.from_numpy(smpl_data["trans"][frames]).float()}
    return smpl_data, duration