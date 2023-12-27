import os
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader
from argparse import Namespace
from utilities3 import *
from PDE_Net import *
from train_utils import *
import gc
import wandb
# wandb.login()

torch.set_default_dtype(torch.float32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ====================================================================== #

config_FNO = Namespace(
project_name = 'FNO',
sdf_file = r'data/allzone_distance.npy',
train_file = r'data/allzone_label.npy',

modes = 12,
width = 40,
activation = 'Sigmoid',
optim_type = 'SGD',
lr = 0.002,
weight_decay = 0.0004,
max_norm = 3.6,
dropout_p = 0,

epochs = 10000,
batch_size = 10,

save_path = r'train_state/FNO_ADF'

)

# ====================================================================== #
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
sdf_data = np.load(config_FNO.sdf_file)
sdf_data = torch.as_tensor(sdf_data).float().to(device)
train_data = np.load(config_FNO.train_file)
train_data = torch.as_tensor(train_data).float().to(device)

sdf_data = sdf_data.permute(0,2,3,1)

x_normalizer = GaussianNormalizer(sdf_data)
x_data = x_normalizer.encode(sdf_data)
y_normalizer = GaussianNormalizer(train_data)
y_data = y_normalizer.encode(train_data)

activation = torch.nn.__dict__[config_FNO.activation]()
model = FNO2d(config_FNO.modes, config_FNO.modes, config_FNO.width, activation, config_FNO.dropout_p).to(device)
optimizer = optim.__dict__[config_FNO.optim_type](model.parameters(), lr = config_FNO.lr, weight_decay = config_FNO.weight_decay)
loss = nn.MSELoss()

train_loader = DataLoader(torch.utils.data.TensorDataset(x_data, y_data), batch_size = config_FNO.batch_size, shuffle = True)

model.train()
for epoch in range(1, config_FNO.epochs+1):
	loss_epoch = 0
	for batch in train_loader:
		x, y = batch
		x, y =x.to(device), y.to(device)

		pred = model(x)
		pred = pred.permute(0,3,1,2)

		loss_out = loss(pred.clone(), y.clone())

		regularization_loss = 0
		for param in model.parameters():
			regularization_loss += torch.norm(param, p=2)
		loss_out += config_FNO.weight_decay * regularization_loss
		loss_epoch += loss_out.item()

		optimizer.zero_grad()
		loss_out.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), config_FNO.max_norm)
		optimizer.step()
	loss_epoch /= len(train_loader)

	print(f'Epoch: {epoch:3d}/{config_FNO.epochs:3d} Loss: {loss_epoch}')
	if os.path.exists(config_FNO.save_path) == False:
		os.mkdir(config_FNO.save_path)
	if epoch % 100 == 0:
		save_path = os.path.join(config_FNO.save_path, '{}_{}.pth'.format(config_FNO.project_name, now_time))
		save_model(save_path, epoch, model, optimizer, now_time)