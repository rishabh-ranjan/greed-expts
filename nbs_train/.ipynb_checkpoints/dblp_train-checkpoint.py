## args: cuda_index train_batch_size lr weight_decay step_up_down


import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '../../pyged/lib')

CUDA_INDEX = int(sys.argv[1])
batch_size = int(sys.argv[2])
lr = float(sys.argv[3])
weight_decay = float(sys.argv[4])
step_size = int(sys.argv[5])

NAME = 'DBLP'
CLASSES = 8

import os
import pickle
import random
import time

import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
import torch.optim
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm

from neuro import config, datasets, metrics, models, train, utils, viz
import pyged

from importlib import reload
reload(config)
reload(datasets)
reload(metrics)
reload(models)
reload(pyged)
reload(train)
reload(utils)
reload(viz)

train_set, train_meta = torch.load(f'../data/{NAME}/train.pt', map_location='cpu')

val_set, _ = torch.load(f'../data/{NAME}/val.pt', map_location='cpu')

model = models.NormSEDModel(8, CLASSES, 64, 64)
model.load_state_dict(torch.load('../runlogs/DBLP/1628504752.911032/best_model.pt', map_location='cpu'))
model = model.to(config.device)

loader = tg.data.DataLoader(list(zip(*train_set)), batch_size=batch_size, shuffle=True)
val_loader = tg.data.DataLoader(list(zip(*val_set)), batch_size=1000, shuffle=True)

dump_path = os.path.join(f'../runlogs/{NAME}', str(time.time()))
os.mkdir(dump_path)
train.train_full(model, loader, val_loader, lr=lr, weight_decay=weight_decay, cycle_patience=10, step_size_up=step_size, step_size_down=step_size, dump_path=dump_path)
