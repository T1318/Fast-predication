import os
import json
from argparse import Namespace
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import vtkmodules.all as vtk
import numpy as np
import operator
from functools import reduce
from functools import partial
from scipy.interpolate import griddata

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
	def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
			super(MatReader, self).__init__()

			self.to_torch = to_torch
			self.to_cuda = to_cuda
			self.to_float = to_float

			self.file_path = file_path

			self.data = None
			self.old_mat = None
			self._load_file()

	def _load_file(self):
			try:
					self.data = scipy.io.loadmat(self.file_path)
					self.old_mat = True
			except:
					self.data = h5py.File(self.file_path)
					self.old_mat = False

	def load_file(self, file_path):
			self.file_path = file_path
			self._load_file()

	def read_field(self, field):
			x = self.data[field]

			if not self.old_mat:
					x = x[()]
					x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

			if self.to_float:
					x = x.astype(np.float32)

			if self.to_torch:
					x = torch.from_numpy(x)

					if self.to_cuda:
							x = x.cuda()

			return x

	def set_cuda(self, to_cuda):
			self.to_cuda = to_cuda

	def set_torch(self, to_torch):
			self.to_torch = to_torch

	def set_float(self, to_float):
			self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
	def __init__(self, x, eps=0.00001):
			super(UnitGaussianNormalizer, self).__init__()

			# x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
			self.mean = torch.mean(x, 0)
			self.std = torch.std(x, 0)
			self.eps = eps

	def encode(self, x):
			x = (x - self.mean) / (self.std + self.eps)
			return x

	def decode(self, x, sample_idx=None):
			if sample_idx is None:
					std = self.std + self.eps # n
					mean = self.mean
			else:
					if len(self.mean.shape) == len(sample_idx[0].shape):
							std = self.std[sample_idx] + self.eps  # batch*n
							mean = self.mean[sample_idx]
					if len(self.mean.shape) > len(sample_idx[0].shape):
							std = self.std[:,sample_idx]+ self.eps # T*batch*n
							mean = self.mean[:,sample_idx]

			# x is in shape of batch*n or T*batch*n
			x = (x * std) + mean
			return x

	def cuda(self):
			self.mean = self.mean.cuda()
			self.std = self.std.cuda()

	def cpu(self):
			self.mean = self.mean.cpu()
			self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
	def __init__(self, x, eps=0.00001):
			super(GaussianNormalizer, self).__init__()

			self.mean = torch.mean(x)
			self.std = torch.std(x)
			self.eps = eps

	def encode(self, x):
			x = (x - self.mean) / (self.std + self.eps)
			return x

	def decode(self, x, sample_idx=None):
			x = (x * (self.std + self.eps)) + self.mean
			return x

	def cuda(self):
			self.mean = self.mean.cuda()
			self.std = self.std.cuda()

	def cpu(self):
			self.mean = self.mean.cpu()
			self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
	def __init__(self, x, low=0.0, high=1.0):
			super(RangeNormalizer, self).__init__()
			mymin = torch.min(x, 0)[0].view(-1)
			mymax = torch.max(x, 0)[0].view(-1)

			self.a = (high - low)/(mymax - mymin)
			self.b = -self.a*mymax + high

	def encode(self, x):
			s = x.size()
			x = x.view(s[0], -1)
			x = self.a*x + self.b
			x = x.view(s)
			return x

	def decode(self, x):
			s = x.size()
			x = x.view(s[0], -1)
			x = (x - self.b)/self.a
			x = x.view(s)
			return x

#loss function with rel/abs Lp loss
class LpLoss(object):
	def __init__(self, d=2, p=2, size_average=True, reduction=True):
			super(LpLoss, self).__init__()

			#Dimension and Lp-norm type are postive
			assert d > 0 and p > 0

			self.d = d
			self.p = p
			self.reduction = reduction
			self.size_average = size_average

	def abs(self, x, y):
			num_examples = x.size()[0]

			#Assume uniform mesh
			h = 1.0 / (x.size()[1] - 1.0)

			all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

			if self.reduction:
					if self.size_average:
							return torch.mean(all_norms)
					else:
							return torch.sum(all_norms)

			return all_norms

	def rel(self, x, y):
			num_examples = x.size()[0]

			diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
			y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

			if self.reduction:
					if self.size_average:
							return torch.mean(diff_norms/y_norms)
					else:
							return torch.sum(diff_norms/y_norms)

			return diff_norms/y_norms

	def __call__(self, x, y):
			return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
		def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
			super(HsLoss, self).__init__()

			#Dimension and Lp-norm type are postive
			assert d > 0 and p > 0

			self.d = d
			self.p = p
			self.k = k
			self.balanced = group
			self.reduction = reduction
			self.size_average = size_average

			if a == None:
					a = [1,] * k
			self.a = a

		def rel(self, x, y):
			num_examples = x.size()[0]
			diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
			y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
			if self.reduction:
					if self.size_average:
							return torch.mean(diff_norms/y_norms)
					else:
							return torch.sum(diff_norms/y_norms)
			return diff_norms/y_norms

		def __call__(self, x, y, a=None):
			nx = x.size()[1]
			ny = x.size()[2]
			k = self.k
			balanced = self.balanced
			a = self.a
			x = x.view(x.shape[0], nx, ny, -1)
			y = y.view(y.shape[0], nx, ny, -1)

			k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
			k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
			k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
			k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

			x = torch.fft.fftn(x, dim=[1, 2])
			y = torch.fft.fftn(y, dim=[1, 2])

			if balanced==False:
					weight = 1
					if k >= 1:
							weight += a[0]**2 * (k_x**2 + k_y**2)
					if k >= 2:
							weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
					weight = torch.sqrt(weight)
					loss = self.rel(x*weight, y*weight)
			else:
					loss = self.rel(x, y)
					if k >= 1:
							weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
							loss += self.rel(x*weight, y*weight)
					if k >= 2:
							weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
							loss += self.rel(x*weight, y*weight)
					loss = loss / (k+1)

			return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
	def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
		super(DenseNet, self).__init__()

		self.n_layers = len(layers) - 1

		assert self.n_layers >= 1

		self.layers = nn.ModuleList()

		for j in range(self.n_layers):
			self.layers.append(nn.Linear(layers[j], layers[j+1]))

			if j != self.n_layers - 1:
				if normalize:
					self.layers.append(nn.BatchNorm1d(layers[j+1]))

				self.layers.append(nonlinearity())

		if out_nonlinearity is not None:
			self.layers.append(out_nonlinearity())

	def forward(self, x):
		for _, l in enumerate(self.layers):
			x = l(x)

		return x


# print the number of parameters
def count_params(model):
	c = 0
	for p in list(model.parameters()):
		c += reduce(operator.mul, list(p.size()))
	return c

# save the model
def save_model(save_path, epoch, model, optimizer, now_time):
	torch.save({'epoch': epoch, 'now_time': now_time, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

# load the model
def load_model(save_path, model, optimizer = None, file_path = None):
	if file_path is None:
		file_list = os.listdir(save_path)
		file_list.sort(key = lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
		last_file = os.path.join(save_path, file_list[-1])
	else:
		last_file = os.path.join(save_path, file_path)

	model_data = torch.load(last_file)
	model.load_state_dict(model_data['model_state_dict'])
	if optimizer is not None:
		optimizer.load_state_dict(model_data['optimizer_state_dict'])
	print('Load model at time:',model_data['now_time'])

# save the dict
def save_dict(save_path, config):
	config.device = str(config.device)
	config.model = str(config.model)
	if hasattr(config,'encoder'):
		config.encoder = str(config.encoder)
	config_dict = vars(config)

	with open(save_path, 'w') as f:
		json.dump(config_dict, f, indent=2)

# load the dict
def load_dict(save_path, file_path = None):
	if file_path is None:
		file_list = os.listdir(save_path)
		file_list.sort(key = lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
		last_file = os.path.join(save_path, file_list[-1])
	else:
		last_file = os.path.join(save_path, file_path)

	with open(last_file, 'r') as f:
		config_dict = json.load(f)
	# config_dict['device'] = torch.device(config_dict['device'])
	# config_dict['model'] = eval(config_dict['model'])
	config = Namespace(**config_dict)
	print('Load model at time:',last_file.split('.')[0])
	return config

# 0-1 normalization
def normalize(data):
	'''
	min:	[batch_size,x,y,variable],		
	max:	[batch_size,x,y,variable],		
	'''
	min = torch.zeros((data.shape[0],data.shape[1]))
	max = torch.zeros((data.shape[0],data.shape[1])) 
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			min[i,j] = torch.min(data[i,:,:,j])
			max[i,j] = torch.max(data[i,:,:,j])
			data[i,:,:,j] = (data[i,:,:,j]-min[i,j])/(max[i,j]-min[i,j]+1e-6)
	return data,min,max

# inverse 0-1 normalization
def inverse_normalize(data,min,max):
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i,:,:,j] = data[i,:,:,j]*(max[i,j]-min[i,j]+1e-6)+min[i,j]
	return data

# create vtk file
def cloud2vtk(x_num,y_num,data,data_name):
	# 创建一个示例的二维速度场，假设是一个 m x n 的矩阵
	m, n = x_num, y_num
	velocity_field = np.random.rand(m, n)  # 使用随机数据作为示例速度场数据

	# 创建一个 VTKImageData 对象
	grid = vtk.vtkImageData()
	grid.SetDimensions(m, n, 1)  # 设置网格尺寸
	grid.SetSpacing(0.01, 0.01, 0.01)  # 设置网格间距

	# 创建一个 VTK 数组来存储速度大小数据
	velocity_data = vtk.vtkDoubleArray()
	velocity_data.SetName("Velocity")  # 设置数组名称
	for i in range(n):
			for j in range(m):
					velocity_data.InsertNextValue(data[i, j])

	# 将速度大小数据附加到 VTKImageData 中
	grid.GetPointData().AddArray(velocity_data)

	# 创建一个 VTK XML Writer 来保存 VTK 文件
	writer = vtk.vtkXMLImageDataWriter()
	writer.SetFileName(data_name)  # 设置文件名
	writer.SetInputData(grid)  # 设置输入数据

	# 写入 VTK 文件
	writer.Write()

	# 输出成功消息
	print("VTK 文件已创建: {}".format(data_name))

# uniform downsample
def uniformDownSample(data, downSamplePositon, method='linear'):
		'''
		data: [x, y, u, v, p]
		downSamplePositon: [x, y]
		'''
		interpolated_uvp = griddata(data[:,0:2], data[:,2:5], downSamplePositon, method=method)
		return interpolated_uvp

def cross_fit(repeat_times, matrix1, matrix2 = None):
	if matrix2 != None and repeat_times == 2:
		result = torch.empty_like(matrix1.repeat(2,1))
		result[0::2] = matrix1
		result[1::2] = matrix2
	
	else:
		result = torch.empty_like(matrix1.repeat(repeat_times,1))
		for i in range(repeat_times):
			result[i::repeat_times] = matrix1

	return result