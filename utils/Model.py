import torch.nn as nn

class PDE_ANN(nn.Module):
	def __init__(self, input_size, hidden_size, hidden_layer, output_size, dropout=0.5):
		super(PDE_ANN, self).__init__()
		self.net = []
		self.dropout = nn.Dropout(p=dropout)

		self.net.append(nn.Linear(input_size, hidden_size))
		self.net.append(nn.LeakyReLU())
		
		for i in range(hidden_layer):
			self.net.append(nn.Linear(hidden_size, hidden_size))
			self.net.append(nn.LeakyReLU())
			self.net.append(self.dropout)
			self.net.append(nn.BatchNorm1d(hidden_size))

		self.net.append(nn.Linear(hidden_size, output_size))
		self.net = nn.ModuleList(self.net)
	def forward(self,x):
		for f in self.net:
			x = f(x)
		return x
	
# class PDE_CNN(nn.Module):
	
	
def weight_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
		nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
		nn.init.constant_(m.bias.data, 0.1)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
		nn.init.constant_(m.bias.data, 0.1)
	elif isinstance(m, nn.BatchNorm3d):
		nn.init.constant_(m.weight.data, 1.0)
		nn.init.constant_(m.bias.data, 0.0)
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
		nn.init.constant_(m.bias.data, 0.0)
	elif isinstance(m, nn.BatchNorm1d):
		nn.init.constant_(m.weight.data, 1.0)
		nn.init.constant_(m.bias.data, 0.0)
	else:
		print('unkown layer:', m)

def get_Net(params):
	if params.net == 'ANN':
		pde_net = PDE_ANN([3,16,32,64,128,64,32,16,4])
	pde_net.apply(weight_init)

	return pde_net
