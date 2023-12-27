import numpy as np
import torch
from torch import nn
from Model import PDE_ANN, weight_init
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================================================================
# ====================================================================================================

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

# ======================================================================================#
# ======================================================================================#

class MLP(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels, activation = nn.Identity()):
		super(MLP, self).__init__()
		self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
		self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
		self.func = activation

	def forward(self, x):
		x = self.mlp1(x)
		x = self.func(x)
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
    
class FNO2d(nn.Module):
		def __init__(self, modes1, modes2, width, activation, dropout_p):
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
				self.func = activation
				self.dropout_p = dropout_p
				self.p = nn.Linear(5, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
				self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
				self.mlp0 = MLP(self.width, self.width, self.width, self.func)
				self.mlp1 = MLP(self.width, self.width, self.width, self.func)
				self.mlp2 = MLP(self.width, self.width, self.width, self.func)
				self.mlp3 = MLP(self.width, self.width, self.width, self.func)
				self.w0 = nn.Conv2d(self.width, self.width, 1)
				self.w1 = nn.Conv2d(self.width, self.width, 1)
				self.w2 = nn.Conv2d(self.width, self.width, 1)
				self.w3 = nn.Conv2d(self.width, self.width, 1)
				self.norm = nn.InstanceNorm2d(self.width)
				self.dropout = nn.Dropout2d(self.dropout_p)
				self.q = MLP(self.width, 3, self.width * 4) # output channel is 1: u(x, y)

		def forward(self, x):
				# x[20,64,64,10] [batchsize, x=64, y=64, t_len=10]
				grid = self.get_grid(x.shape, x.device)	# [20,64,64,2]
				x = torch.cat((x, grid), dim=-1)				# [20,64,64,5]
				x = self.p(x)														# p: [20,64,64,5] -> [20,64,64,width=20]
				x = x.permute(0, 3, 1, 2)								# [20,64,64,width=20] -> [20,width=20,64,64] [batchsize, channel, x, y]
				x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

				x1 = self.norm(self.conv0(self.norm(x)))	# Fourier Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x1 = self.mlp0(x1)												# MLP Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x2 = self.w0(x)														# W convolution Layer [20,width=20,64,64] -> [20,width=20,64,64]
				x = x1 + x2
				x = self.func(x)													# Activation Function

				x1 = self.norm(self.conv1(self.norm(x)))
				x1 = self.mlp1(x1)
				x2 = self.w1(x)
				x = x1 + x2
				x = self.func(x)													# Activation Function

				x1 = self.norm(self.conv2(self.norm(x)))
				x1 = self.mlp2(x1)
				x2 = self.w2(x)
				x = x1 + x2
				x = self.func(x)													# Activation Function

				x1 = self.norm(self.conv3(self.norm(x)))
				x1 = self.mlp3(x1)
				x2 = self.w3(x)
				x = x1 + x2

				x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
				x = self.q(x)								# MLP Layer [20,width=20,64,64] -> [20,1,64,64]
				x = self.dropout(x)						# Dropout Layer
				x = x.permute(0, 2, 3, 1)	# [20,width=1,64,64] -> [20,64,64,1]
				return x

		
		def get_grid(self, shape, device):
				batchsize, size_x, size_y = shape[0], shape[1], shape[2]
				gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
				gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
				gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
				gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
				return torch.cat((gridx, gridy), dim=-1).to(device)

# ======================================================================================#
# ======================================================================================#

# 定义编码器
class Encoder(nn.Module):
	def __init__(self, config, input_size, hidden_channel, output_size, layers, activation, dropout_p):
		super().__init__()
		### Convolutional section
		self.config = config

		linear1_inlet = int(hidden_channel*((config.data_x/2**layers) * (config.data_y/2**layers)))
		linear2_inlet = int(hidden_channel*((config.data_x/2**layers) * (config.data_y/2**layers) /2))
		linear3_inlet = int(hidden_channel*output_size)
		linear4_inlet = int(output_size)

		input_channel = input_size
		encoder_cnn = []
		for i in range(layers):
			encoder_cnn.append(nn.Conv2d(input_channel, hidden_channel, 3, stride=1, padding=1))
			encoder_cnn.append(nn.BatchNorm2d(hidden_channel))
			encoder_cnn.append(activation())
			encoder_cnn.append(nn.MaxPool2d(2, stride=2))
			input_channel = hidden_channel
			# hidden_channel *= 2

		self.encoder_cnn = nn.Sequential(*encoder_cnn)
		'''
		self.encoder_cnn = nn.Sequential(
			nn.Conv2d(input_size, hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(hidden_channel),
			activation(True),
			nn.MaxPool2d(2, stride=2),																																									# 1024->512, 384->192
			nn.Conv2d(hidden_channel, 2*hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(2*hidden_channel),
			activation(True),
			nn.MaxPool2d(2, stride=2),																																									# 512->256, 192->96
			nn.Conv2d(2*hidden_channel, 4*hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(4*hidden_channel),
			activation(True),
			nn.MaxPool2d(2, stride=2),																																									# 256->128, 96->48
			nn.Conv2d(4*hidden_channel, 8*hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(8*hidden_channel),
			activation(True),
			nn.MaxPool2d(2, stride=2),																																									# 128->64, 48->24
		)
		'''
		### Flatten layer
		self.flatten = nn.Flatten(start_dim=1)
		### Linear section
		self.encoder_lin = nn.Sequential(
			nn.Linear(linear1_inlet, linear2_inlet),
			nn.BatchNorm1d(linear2_inlet),
			activation(),
			nn.Linear(linear2_inlet, linear3_inlet),
			nn.BatchNorm1d(linear3_inlet),
			activation(),
			nn.Linear(linear3_inlet, linear4_inlet),
		)
	def forward(self, x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		x = self.encoder_lin(x)
		return x

# 定义解码器
class Decoder(nn.Module):
	def __init__(self, config, input_size, hidden_channel, output_size, layers, activation, dropout_p):
		super().__init__()

		linear4_inlet = int(hidden_channel*((config.data_x/2**layers) * (config.data_y/2**layers)))
		linear3_inlet = int(hidden_channel*((config.data_x/2**layers) * (config.data_y/2**layers) /2))
		linear2_inlet = int(hidden_channel*input_size)
		linear1_inlet = int(input_size)

		self.decoder_lin = nn.Sequential(
			nn.Linear(linear1_inlet, linear2_inlet),
			nn.BatchNorm1d(linear2_inlet),
			activation(),
			nn.Linear(linear2_inlet, linear3_inlet),
			nn.BatchNorm1d(linear3_inlet),
			activation(),
			nn.Linear(linear3_inlet, linear4_inlet),
		)

		self.unflatten = nn.Unflatten(dim=1,unflattened_size=(hidden_channel, int(config.data_x/2**layers), int(config.data_y/2**layers)))

		decoder_cnn = []
		for _ in range(layers-1):
			# 添加解卷积层
			decoder_cnn.append(nn.ConvTranspose2d(hidden_channel, hidden_channel, 3, stride=2, padding=1, output_padding=1))
			decoder_cnn.append(nn.Conv2d(hidden_channel, hidden_channel, 3, stride=1, padding=1))
			decoder_cnn.append(nn.BatchNorm2d(hidden_channel))
			decoder_cnn.append(activation())
			# in_channels //= 2  # 减半通道数
		# 添加最后的输出层
		decoder_cnn.append(nn.ConvTranspose2d(hidden_channel, output_size, 3, stride=2, padding=1, output_padding=1))
		decoder_cnn.append(nn.Conv2d(output_size, output_size, 3, stride=1, padding=1))
		
		self.decoder_conv = nn.Sequential(*decoder_cnn)
		'''
		self.decoder_conv = nn.Sequential( 
			nn.ConvTranspose2d(8*hidden_channel, 4*hidden_channel, 3,stride=2,padding=1, output_padding=1),					# 64->128
			nn.Conv2d(4*hidden_channel, 4*hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(4*hidden_channel),
			activation(True),
			nn.ConvTranspose2d(4*hidden_channel, 2*hidden_channel, 3, stride=2,padding=1, output_padding=1),							# 128->256
			nn.Conv2d(2*hidden_channel, 2*hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(2*hidden_channel),
			activation(True),
			nn.ConvTranspose2d(2*hidden_channel, hidden_channel, 3, stride=2,padding=1, output_padding=1),								# 256->512
			nn.Conv2d(hidden_channel, hidden_channel, 3, stride=1, padding=1),
			nn.BatchNorm2d(hidden_channel),
			activation(True),
			nn.ConvTranspose2d(hidden_channel, output_size, 3, stride=2,padding=1, output_padding=1),																				# 512->1024
			nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
		)
		'''
	def forward(self, x):
		x = self.decoder_lin(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		return x

# 定义卷积自动编码器
class ConvAutoencoder(nn.Module):
	def __init__(self,config):
			super(ConvAutoencoder, self).__init__()
			self.config = config
			self.activation = torch.nn.__dict__[self.config.activation]
			self.encoder = Encoder(self.config, self.config.input_size, self.config.hidden_channel, self.config.latent_size, self.config.layers, self.activation, self.config.dropout_p)
			self.decoder = Decoder(self.config, self.config.latent_size, self.config.hidden_channel, self.config.input_size, self.config.layers, self.activation, self.config.dropout_p)

	def forward(self, x):
			x = self.encoder(x)
			x = self.decoder(x)
			return x

# 定义CAE_FLU
class CAE_FLU(nn.Module):
	def __init__(self,config):
		super(CAE_FLU, self).__init__()
		self.config = config
		self.encoder = self.config.encoder
		self.activation = torch.nn.__dict__[self.config.activation]
		self.SDF_net = Encoder(self.config, self.config.sdf_input_size, self.config.hidden_channel, self.config.output_size, self.config.sdf_layers, self.activation, self.config.dropout_p)
		self.main_net = Decoder(self.config, 2*self.config.output_size, self.config.main_hidden_channel, self.config.variables_num, self.config.main_layers, self.activation, self.config.dropout_p)

	def forward(self, typical_case, sdf):
		n = int(typical_case.shape[0])
		m = int(sdf.shape[0])
		repeat_times = int(m/n)

		self.laten_uvp = self.encoder(typical_case)
		self.laten_sdf = self.SDF_net(sdf)

		# 1. [1,L0], [n,L0]-> [n,2L0]
		self.laten_uvp_repeat = cross_fit(repeat_times, self.laten_uvp)
		self.laten = torch.cat((self.laten_uvp_repeat, self.laten_sdf), 1)
		# 2. [1,L0], [n,L0]-> [n,L0]
		# self.laten = self.laten_sdf * self.laten_uvp
		# 3. [1,L0], [n,3,x,y]-> [n,3,x,y]
		# self.laten_uvp = self.encoder(typical_case)
		# self.laten_uvp = self.SDF_net(self.laten_uvp)
		# self.laten = self.uvp * self.laten_sdf

		self.out = self.main_net(self.laten)
		return self.out

if __name__ == '__main__':
	model = DeepONet_NS([3,16,32,64,128,64,32,16,4], [3,16,32,64,128,64,32,16,4])
	torch.onnx.export(model, (torch.randn(1,3), torch.randn(1,3)), "DeepONet_NS.onnx", verbose=True, input_names=['x_branch', 'x_trunk'], output_names=['y_pred'])