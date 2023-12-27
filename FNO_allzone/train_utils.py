import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import Namespace
from utilities3 import *
from PDE_Net import *

torch.set_default_dtype(torch.float32)

def creat_dataloader(config):
	if config.project_name == 'CAE_autoconder':
		label = torch.Tensor(np.load(config.train_file))

		if config.train_ratio:
			train_size = int(len(label) * config.train_ratio)
			train_lable = label[:train_size]
			test_label = label[train_size:]

			data_train, min_train, max_train = normalize(train_lable)
			data_test, min_test, max_test = normalize(test_label)
			train_loader = DataLoader(torch.utils.data.TensorDataset(data_train, min_train, max_train), batch_size = config.batch_size, shuffle = True, drop_last = True)
			test_loader = DataLoader(torch.utils.data.TensorDataset(data_test, min_test, max_test), batch_size = config.batch_size, shuffle = True, drop_last = True)
			return train_loader, test_loader

		else:
			train_label = label
			data_train, min_train, max_train = normalize(train_label)

			train_loader = DataLoader(torch.utils.data.TensorDataset(data_train, min_train, max_train), batch_size = config.batch_size, shuffle = True, drop_last = True)

			return train_loader
	
	#========================================#

	elif config.project_name == 'CAE_FLU':
		sdf = torch.as_tensor(np.load(config.sdf_file)).float()
		label = torch.as_tensor(np.load(config.train_file)).float()

		if config.train_ratio:
			train_size = int(len(label) * config.train_ratio)
			train_sdf = sdf[:train_size]
			test_sdf = sdf[train_size:]
			train_lable = label[:train_size]
			test_label = label[train_size:]

			sdf_train, sdf_min_train, sdf_max_train = normalize(train_sdf)
			sdf_test, sdf_min_test, sdf_max_test = normalize(test_sdf)
			label_train, label_min_train, label_max_train = normalize(train_lable)
			label_test, label_min_test, label_max_test = normalize(test_label)

			train_loader = DataLoader(torch.utils.data.TensorDataset(sdf_train, label_train), batch_size = config.batch_size, shuffle = True, drop_last = True)
			test_loader = DataLoader(torch.utils.data.TensorDataset(sdf_test, label_test), batch_size = config.batch_size, shuffle = True, drop_last = True)
			return train_loader, test_loader
		
		else:
			train_sdf = sdf
			train_label = label

			sdf_train, sdf_min_train, sdf_max_train = normalize(train_sdf)
			label_train, label_min_train, label_max_train = normalize(train_lable)

			train_loader = DataLoader(torch.utils.data.TensorDataset(sdf_train, label_train), batch_size = config.batch_size, shuffle = True, drop_last = True)

			return train_loader


def train_epoch(config, model, optimizer, loss, train_loader):
	if config.project_name == 'CAE_autoconder':

		model.train()
		train_loss_per_epoch = 0
		for batch in train_loader:
			loss_per_batch = 0
			x, min, max = batch

			x = x.to(config.device)

			pred = model(x)

			loss_per_batch = loss(pred.clone(), x.clone())
			regularization_loss = 0
			for param in model.parameters():
				regularization_loss += torch.norm(param, p=2)
			loss_per_batch += config.weight_decay * regularization_loss

			train_loss_per_epoch += loss_per_batch.item()

			optimizer.zero_grad()
			loss_per_batch.backward(retain_graph = True)
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
			optimizer.step()

		train_loss_per_epoch /= len(train_loader)

	#========================================#
	
	elif config.project_name == 'CAE_FLU':
		model.train()
		train_loss_per_epoch = 0
		typical_case = np.load(config.typical_case_file)
		for batch in train_loader:
			loss_per_batch = 0
			sdf, label = batch

			sdf = sdf.to(config.device)
			label = label.to(config.device)
			typical_case = torch.as_tensor(typical_case).float().to(config.device)

			pred = model(typical_case, sdf)

			loss_per_batch = loss(pred.clone(), label.clone())
			regularization_loss = 0
			for param in model.parameters():
				regularization_loss += torch.norm(param, p=2)
			loss_per_batch += config.weight_decay * regularization_loss

			train_loss_per_epoch += loss_per_batch.item()

			optimizer.zero_grad()
			loss_per_batch.backward(retain_graph = True)
			torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
			optimizer.step()

		train_loss_per_epoch /= len(train_loader)

	return train_loss_per_epoch

def eval_epoch(config, model, optimizer, loss, test_loader):
	if config.project_name == 'CAE_autoconder':
		model.eval()
		test_loss_per_epoch = 0
		with torch.no_grad():
			for batch in test_loader:
				loss_per_batch = 0
				x, min, max = batch

				x = x.to(config.device)

				pred = model(x)

				loss_per_batch = loss(pred.clone(), x.clone())
				regularization_loss = 0
				for param in model.parameters():
					regularization_loss += torch.norm(param, p=2)
				loss_per_batch += config.weight_decay * regularization_loss

				test_loss_per_epoch += loss_per_batch.item()

		test_loss_per_epoch /= len(test_loader)
	
	elif config.project_name == 'CAE_FLU':
		model.eval()
		test_loss_per_epoch = 0
		typical_case = np.load(config.typical_case_file)
		with torch.no_grad():
			for batch in test_loader:
				loss_per_batch = 0
				sdf, label = batch

				sdf = sdf.to(config.device)
				label = label.to(config.device)
				typical_case = torch.as_tensor(typical_case).float().to(config.device)

				pred = model(typical_case, sdf)

				loss_per_batch = loss(pred.clone(), label.clone())
				regularization_loss = 0
				for param in model.parameters():
					regularization_loss += torch.norm(param, p=2)
				loss_per_batch += config.weight_decay * regularization_loss

				test_loss_per_epoch += loss_per_batch.item()

			test_loss_per_epoch /= len(test_loader)

	return test_loss_per_epoch

def train(config):
	model = config.model(config).to(config.device)
	model.apply(weight_init)

	optimizer = torch.optim.__dict__[config.optimizer](params=model.parameters(), lr=config.lr)
	loss = nn.MSELoss()

	if config.continue_training:
		load_model(config.train_state_dir, model, optimizer)
		
	#================
	now_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
	#================

	if not os.path.exists(config.train_state_dir):
			os.makedirs(config.train_state_dir)
			
	#================
	wandb.init(project=config.project_name, config=config.__dict__, name=now_time, save_code=True, mode='offline')
	model.run_id = wandb.run.id
	#================

	for epoch in range(config.epochs+1):
		if config.train_ratio:
			train_loader, test_loader = creat_dataloader(config)	# [Batch, x, y, variables]
			train_loss_per_epoch = train_epoch(config, model, optimizer, loss, train_loader)
			test_loss_per_epoch = eval_epoch(config, model, optimizer, loss, test_loader)
			print('{}, Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(now_time, epoch, train_loss_per_epoch, test_loss_per_epoch))
			wandb.log({'epoch': epoch, 'train_loss': train_loss_per_epoch, 'test_loss': test_loss_per_epoch})

		
		else:
			train_loader = creat_dataloader(config)	# [Batch, x, y, variables]
			train_loss_per_epoch = train_epoch(config, model, optimizer, loss, train_loader)
			print('{}, Epoch: {}, Train Loss: {:.4f}'.format(now_time, epoch, train_loss_per_epoch))
			wandb.log({'epoch': epoch, 'train_loss': train_loss_per_epoch})

		if epoch % 100 == 0:
			save_path = os.path.join(config.train_state_dir, '{}_{}.pth'.format(config.project_name, now_time))
			save_model(save_path, epoch, model, optimizer, now_time)

	return model, optimizer
	
def train_sweep():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	with wandb.init(mode='offline'):
		# if wandb.config.project_name == 'CAE_Autoconder':
		model = ConvAutoencoder(wandb.config).to(device)
		# else:
		# 	model = CAE_FLU(wandb.config).to(device)
		model.apply(weight_init)

		optimizer = torch.optim.__dict__[wandb.config.optimizer](params=model.parameters(), lr=wandb.config.lr)
		loss = nn.MSELoss()

		if wandb.config.continue_training:
			load_model(wandb.config.train_state_dir, model, optimizer)
			
		#================
		now_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
		#================

		if not os.path.exists(wandb.config.train_state_dir):
				os.makedirs(wandb.config.train_state_dir)

		for epoch in range(wandb.config.epochs+1):
			if wandb.config.train_ratio:
				train_loader, test_loader = creat_dataloader(wandb.config)	# [Batch, x, y, variables]
				train_loss_per_epoch = train_epoch(wandb.config, model, optimizer, loss, train_loader)
				test_loss_per_epoch = eval_epoch(wandb.config, model, optimizer, loss, test_loader)
				print('{}, Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(now_time, epoch, train_loss_per_epoch, test_loss_per_epoch))
				wandb.log({'epoch': epoch, 'train_loss': train_loss_per_epoch, 'test_loss': test_loss_per_epoch})
			
			else:
				train_loader = creat_dataloader(wandb.config)	# [Batch, x, y, variables]
				train_loss_per_epoch = train_epoch(wandb.config, model, optimizer, loss, train_loader)
				print('{}, Epoch: {}, Train Loss: {:.4f}'.format(now_time, epoch, train_loss_per_epoch))
				wandb.log({'epoch': epoch, 'train_loss': train_loss_per_epoch})

			if epoch % 100 == 0:
				save_path = os.path.join(wandb.config.train_state_dir, '{}_{}.pth'.format(wandb.config.project_name, now_time))
				save_model(save_path, epoch, model, optimizer, now_time)
		wandb.finish()
	return model, optimizer