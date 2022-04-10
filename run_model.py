import argparse
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np
import tqdm
import logging

from cswm import utils
from cswm.models.modules import RewardPredictor, CausalTransitionModel
from cswm.utils import OneHot, PathDataset

import sys
import datetime
import os
import cv2

from itertools import chain

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')
args_eval = parser.parse_args()


meta_file = args_eval.save_folder / 'metadata.pkl'
model_file = args_eval.save_folder / 'finetuned_model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')

input_shape = (3, 50, 50)

dataset = utils.StateTransitionsDataset(
   hdf5_file=args.dataset, action_transform=OneHot(args.action_dim))

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model = CausalTransitionModel(
    embedding_dim_per_object=args.embedding_dim_per_object,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    input_shape=input_shape,
    modular=args.modular,
    predict_diff=args.predict_diff,
    vae=args.vae,
    num_objects=args.num_objects,
    encoder=args.encoder,
    gnn=args.gnn,
    multiplier=args.multiplier,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action).to(device)

model.load_state_dict(torch.load(model_file))

for batch_idx, data_batch in enumerate(iterator):
    model.train()
    data_batch = [tensor.to(device) for tensor in data_batch]
    obs, action, next_obs, _, _ = data_batch

    state, _ = model.encode(obs)
    next_state = model.transition(state, actions)


