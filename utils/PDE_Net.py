import numpy as np
import torch
from torch import nn
from utils.Model import PDE_ANN, weight_init
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepONet(nn.Module):
	
	def __init__(self, branch_layer, trunk_layer):
		super(DeepONet_NS, self).__init__()
		self.branch_layer = branch_layer
		self.trunk_layer = trunk_layer

		self.loss_fun = nn.MSELoss()

		self.branch_net = PDE_ANN(self.branch_layer)
		self.trunk_net = PDE_ANN(self.trunk_layer)

		self.bias_last = torch.tensor(torch.zeros(1), requires_grad=True)

	def forward(self, x_branch, x_trunk):
		y_branch = self.branch_net(x_branch)
		y_trunk = self.trunk_net(x_trunk)
		# Dot product
		if y_branch.shape[1] != y_trunk.shape[1]:
			raise AssertionError('The dimension of y_branch and y_trunk should be the same.')
		y = torch.einsum('ik,jk->ij', y_branch, y_trunk)
		# Add bias
		y = y + self.bias_last
		return y
	
	def loss(self, x_branch, x_trunk, y_true):
		y_pred = self.forward(x_branch, x_trunk)
		loss = self.loss_fun(y_pred, y_true)
		return loss
	
class DeepONet_NS(nn.Module):
	
	def __init__(self, branch_layer, trunk_layer):
		super(DeepONet_NS, self).__init__()
		self.branch_layer = branch_layer
		self.trunk_layer = trunk_layer

		self.loss_fun = nn.MSELoss()

		self.branch_net = PDE_ANN(self.branch_layer)
		self.trunk_net = PDE_ANN(self.trunk_layer)

		self.bias_last = torch.tensor(torch.zeros(1), requires_grad=True, device=device)

	def forward(self, x_branch, x_trunk):
		y_branch = self.branch_net(x_branch)
		y_trunk = self.trunk_net(x_trunk)
		# Dot product
		if y_branch.shape[1] != y_trunk.shape[1]:
			raise AssertionError('The dimension of y_branch and y_trunk should be the same.')
		y = torch.einsum('ik,jk->ij', y_branch, y_trunk)
		# Add bias
		y = y + self.bias_last
		return y
	
	def loss(self, x_branch, x_trunk, y_true):
		y_pred = self.forward(x_branch, x_trunk)
		loss = self.loss_fun(y_pred, y_true)
		return loss
	
class MLP(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels):
		super(MLP, self).__init__()
		self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
		self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

	def forward(self, x):
		x = self.mlp1(x)
		x = F.gelu(x)
		x = self.mlp2(x)
		return x

class SpectralConv2d(nn.Module):
		def __init__(self, in_channels, out_channels, modes1, modes2):
				super(SpectralConv2d, self).__init__()

				"""
				2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
				"""

				self.in_channels = in_channels
				self.out_channels = out_channels
				self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
				self.modes2 = modes2

				self.scale = (1 / (in_channels * out_channels))
				self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
				self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

		# Complex multiplication
		def compl_mul2d(self, input, weights):
				# (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
				return torch.einsum("bixy,ioxy->boxy", input, weights)

		def forward(self, x):
				batchsize = x.shape[0]
				#Compute Fourier coeffcients up to factor of e^(- something constant)
				x_ft = torch.fft.rfft2(x)

				# Multiply relevant Fourier modes
				out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
				out_ft[:, :, :self.modes1, :self.modes2] = \
						self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
				out_ft[:, :, -self.modes1:, :self.modes2] = \
						self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

				#Return to physical space
				x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
				return x
class MLP(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels):
		super(MLP, self).__init__()
		self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
		self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

	def forward(self, x):
		x = self.mlp1(x)
		x = F.gelu(x)
		x = self.mlp2(x)
		return x
    
class FNO2d(nn.Module):
		def __init__(self, modes1, modes2,  width):
				super(FNO2d, self).__init__()

				"""
				The overall network. It contains 4 layers of the Fourier layer.
				1. Lift the input to the desire channel dimension by self.fc0 .
				2. 4 layers of the integral operators u' = (W + K)(u).
						W defined by self.w; K defined by self.conv .
				3. Project from the channel space to the output space by self.fc1 and self.fc2 .
				
				input: the solution of the coefficient function and locations (a(x, y), x, y)
				input shape: (batchsize, x=s, y=s, c=3)
				output: the solution 
				output shape: (batchsize, x=s, y=s, c=1)
				"""

				self.modes1 = modes1	# modes1 = 12
				self.modes2 = modes2	# modes2 = 12
				self.width = width		# width = 20
				self.padding = 8 # pad the domain if input is non-periodic

				self.p = nn.Linear(5, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
				self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.mlp0 = MLP(self.width, self.width, self.width)
				self.mlp1 = MLP(self.width, self.width, self.width)
				self.mlp2 = MLP(self.width, self.width, self.width)
				self.mlp3 = MLP(self.width, self.width, self.width)
				self.w0 = nn.Conv2d(self.width, self.width, 1)
				self.w1 = nn.Conv2d(self.width, self.width, 1)
				self.w2 = nn.Conv2d(self.width, self.width, 1)
				self.w3 = nn.Conv2d(self.width, self.width, 1)
				self.norm = nn.InstanceNorm2d(self.width)
				self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

		def forward(self, x):
				# x[20,64,64,10] [batchsize, x=64, y=64, t_len=10]
				grid = self.get_grid(x.shape, x.device)	# [20,64,64,2]
				x = torch.cat((x, grid), dim=-1)				# [20,64,64,12]
				x = self.p(x)														# p: [20,64,64,12] -> [20,64,64,width=20]
				x = x.permute(0, 3, 1, 2)								# [20,64,64,width=20] -> [20,width=20,64,64] [batchsize, channel, x, y]
				x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

				x1 = self.norm(self.conv0(self.norm(x)))	# Fourier Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x1 = self.mlp0(x1)												# MLP Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x2 = self.w0(x)														# W convolution Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x = x1 + x2
				x = F.gelu(x)

				x1 = self.norm(self.conv1(self.norm(x)))
				x1 = self.mlp1(x1)
				x2 = self.w1(x)
				x = x1 + x2
				x = F.gelu(x)

				x1 = self.norm(self.conv2(self.norm(x)))
				x1 = self.mlp2(x1)
				x2 = self.w2(x)
				x = x1 + x2
				x = F.gelu(x)

				x1 = self.norm(self.conv3(self.norm(x)))
				x1 = self.mlp3(x1)
				x2 = self.w3(x)
				x = x1 + x2

				x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
				x = self.q(x)								# MLP Layer [20,width=20,64,64] -> [20,1,64,64]
				x = x.permute(0, 2, 3, 1)	# [20,width=1,64,64] -> [20,64,64,1]
				return x

		
		def get_grid(self, shape, device):
				batchsize, size_x, size_y = shape[0], shape[1], shape[2]
				gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
				gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
				gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
				gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
				return torch.cat((gridx, gridy), dim=-1).to(device)


if __name__ == '__main__':
	model = DeepONet_NS([3,16,32,64,128,64,32,16,4], [3,16,32,64,128,64,32,16,4])
	torch.onnx.export(model, (torch.randn(1,3), torch.randn(1,3)), "DeepONet_NS.onnx", verbose=True, input_names=['x_branch', 'x_trunk'], output_names=['y_pred'])