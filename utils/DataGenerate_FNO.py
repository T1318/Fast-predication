import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
'''
fluent导出文件为: x y p u v
经过处理后文件为: x y u v p
'''
def get_file(dir):
		'''
		把瞬态文件的多个file合并成一个list
		'''
		file_list = []
		for root, dirs, files in os.walk(dir):
			for file in files:
				file_list.append(os.path.join(root, file))
		return file_list
		
def Normalize(x):
	eps = 1e-6
	mean = torch.zeros((x.shape[0],x.shape[1],x.shape[-1]))
	std = torch.zeros((x.shape[0],x.shape[1],x.shape[-1]))
	for i in range(x.shape[0]):
		for t in range(x.shape[1]):
			for j in range(x.shape[-1]):
				mean[i,t,j] = torch.mean(x[i,t,:,:,j])
				std[i,t,j] = torch.std(x[i,t,:,:,j])
				x[i,t,:,:,j] = (x[i,t,:,:,j]-mean[i,t,j])/(std[i,t,j]+eps)
	return x, mean, std

def InNormalize(x, mean, std):
	eps = 1e-6
	for i in range(x.shape[0]):
			for j in range(x.shape[-1]):
				x[i,:,:,j] = x[i,:,:,j]*(std[i,j]+eps)+mean[i,j]
	return x

def enhance_data(datafile, is_pred=True, split=4):
	# split
	x = np.round(datafile[:,0],2)
	y = np.round(datafile[:,1],2)
	p = datafile[:,2]
	u = datafile[:,3]
	v = datafile[:,4]

	# set the index
	eps = 1e-6
	indices = [[np.where((i/2-eps<=datafile[:,0]) & (datafile[:,0]<=(i+1)/2+eps) & (j/2-eps<=datafile[:,1]) & (datafile[:,1]<=(j+1)/2+eps)) for j in range(6)] for i in range(16)]		# 8*3*10201
	# 将数据分为8*3*10201的数组
	xx = np.zeros((16,6,2601))
	yy = np.zeros((16,6,2601))
	pp = np.zeros((16,6,2601))
	uu = np.zeros((16,6,2601))
	vv = np.zeros((16,6,2601))
	for i in range(16):
		for j in range(6):
			if 8<=i<=9 and 0<=j<=1:
				continue
			xx[i,j,:] = x[indices[i][j][0]]
			yy[i,j,:] = y[indices[i][j][0]]
			pp[i,j,:] = p[indices[i][j][0]]
			uu[i,j,:] = u[indices[i][j][0]]
			vv[i,j,:] = v[indices[i][j][0]]
	xx = xx.reshape(16,6,51,51)
	yy = yy.reshape(16,6,51,51)
	pp = pp.reshape(16,6,51,51)
	uu = uu.reshape(16,6,51,51)
	vv = vv.reshape(16,6,51,51)
	# 现在有8*3*51*51的uvp数据，需要对其数据强化
	x_original = np.empty((0, 51, 51))
	y_original = np.empty((0, 51, 51))
	u_original = np.empty((0, 51, 51))
	v_original = np.empty((0, 51, 51))
	p_original = np.empty((0, 51, 51))
	if is_pred:
		'''
		如果是用来预测，将16*6*51*51的数据按顺序整理成96*51*51的数据
		'''
		for i in range(16):
			for j in range(6):
				x_original = np.vstack((x_original,xx[i,j,:,:].reshape(1,51,51)))
				y_original = np.vstack((y_original,yy[i,j,:,:].reshape(1,51,51)))
				u_original = np.vstack((u_original,uu[i,j,:,:].reshape(1,51,51)))
				v_original = np.vstack((v_original,vv[i,j,:,:].reshape(1,51,51)))
				p_original = np.vstack((p_original,pp[i,j,:,:].reshape(1,51,51)))

	else:
		'''
		如果是用来训练，将16*6*51*51的数据强化，一个图分split进行组合。
		'''
		for i in range(16-1):
			for j in range(6-1):
				for m in range(split):
					for n in range(split):
						x_temp0 = xx[i,j,m*51//split:,n*51//split:]
						x_temp1 = xx[i+1,j,m*51//split:,:n*51//split]
						x_temp2 = xx[i,j+1,:m*51//split,n*51//split:]
						x_temp3 = xx[i+1,j+1,:m*51//split,:n*51//split]
						x_temp01 = np.concatenate((x_temp0,x_temp1),axis = 1)
						x_temp23 = np.concatenate((x_temp2,x_temp3),axis = 1)
						x_temp = np.concatenate((x_temp01,x_temp23),axis = 0).reshape(1,51,51)
						x_original = np.vstack((x_original,x_temp))

						y_temp0 = yy[i,j,m*51//split:,n*51//split:]
						y_temp1 = yy[i+1,j,m*51//split:,:n*51//split]
						y_temp2 = yy[i,j+1,:m*51//split,n*51//split:]
						y_temp3 = yy[i+1,j+1,:m*51//split,:n*51//split]
						y_temp01 = np.concatenate((y_temp0,y_temp1),axis = 1)
						y_temp23 = np.concatenate((y_temp2,y_temp3),axis = 1)
						y_temp = np.concatenate((y_temp01,y_temp23),axis = 0).reshape(1,51,51)
						y_original = np.vstack((y_original,y_temp))

						u_temp0 = uu[i,j,m*51//split:,n*51//split:]
						u_temp1 = uu[i+1,j,m*51//split:,:n*51//split]
						u_temp2 = uu[i,j+1,:m*51//split,n*51//split:]
						u_temp3 = uu[i+1,j+1,:m*51//split,:n*51//split]
						u_temp01 = np.concatenate((u_temp0,u_temp1),axis = 1)
						u_temp23 = np.concatenate((u_temp2,u_temp3),axis = 1)
						u_temp = np.concatenate((u_temp01,u_temp23),axis = 0).reshape(1,51,51)
						u_original = np.vstack((u_original,u_temp))

						v_temp0 = vv[i,j,m*51//split:,n*51//split:]
						v_temp1 = vv[i+1,j,m*51//split:,:n*51//split]
						v_temp2 = vv[i,j+1,:m*51//split,n*51//split:]
						v_temp3 = vv[i+1,j+1,:m*51//split,:n*51//split]
						v_temp01 = np.concatenate((v_temp0,v_temp1),axis = 1)
						v_temp23 = np.concatenate((v_temp2,v_temp3),axis = 1)
						v_temp = np.concatenate((v_temp01,v_temp23),axis = 0).reshape(1,51,51)
						v_original = np.vstack((v_original,v_temp))

						p_temp0 = pp[i,j,m*51//split:,n*51//split:]
						p_temp1 = pp[i+1,j,m*51//split:,:n*51//split]
						p_temp2 = pp[i,j+1,:m*51//split,n*51//split:]
						p_temp3 = pp[i+1,j+1,:m*51//split,:n*51//split]
						p_temp01 = np.concatenate((p_temp0,p_temp1),axis = 1)
						p_temp23 = np.concatenate((p_temp2,p_temp3),axis = 1)
						p_temp = np.concatenate((p_temp01,p_temp23),axis = 0).reshape(1,51,51)
						p_original = np.vstack((p_original,p_temp))

	variabl = np.stack((u_original,v_original,p_original),axis=3)

	# coordinate = np.stack((x_original,y_original),axis=3)
	'''
	variabl:		[batch_size,51,51,3]
	coordinate:	[batch_size,51,51,2]
	'''
	return variabl
	
class Dataset_FNO(Dataset):
	def __init__(self,datafile_dir,is_pred=False,split=4,is_save=False,datafile=None):
		self.is_pred = is_pred
		self.split = split
		self.interval = 100
		if datafile is not None:	# self.ratio,self.is_pred,self.time_step,self.length,self.variable,self.coordinate,self.bc,self.mean,self.std

			datafile = h5.File(datafile,'r')
			self.x = np.array(datafile['x'])
			self.y = np.array(datafile['y'])
			self.time_step = self.x.shape[0]
			self.length = self.x.shape[1]
		else:
			file_list = get_file(datafile_dir)
			temp_file = np.loadtxt(file_list[0])
			var = enhance_data(temp_file, is_pred=self.is_pred, split=self.split)
			self.time_step = (len(file_list)-1)//self.interval
			self.length = var.shape[0]

			# 将瞬态数据存在第一个维度 [time_step,batch_size,51,51,3]
			self.variable = np.zeros((self.time_step+1,self.length,51,51,3))
			for i in range(self.time_step+1):
				datafile = np.loadtxt(file_list[i*self.interval])
				print('Finish: ',file_list[i*self.interval])
				self.variable[i] = enhance_data(datafile, is_pred=self.is_pred, split=self.split)

			# 边界和数据错位，上一刻边界用来预测下一刻数据
			self.x = self.variable[:-1]
			self.y = self.variable[1:]

			if is_save:
				save_path = os.path.join(datafile_dir,'train_data.h5')
				self.save_data(save_path)
				print('Finish: ',save_path)

		'''reshape之前数据形状:
		x: [time_step,length,51,51,3]
		y: [time_step,length,51,51,3]
		coordinate: [time_step,length,51,51,2]
		mean: [time_step,length,3]
		std: [time_step,length,3]
		'''
	def __len__(self):
		return self.length
	def save_data(self,save_path):
		file = h5.File(save_path,'w')
		file.create_dataset('x',data=self.x)
		file.create_dataset('y',data=self.y)
		file.close()
	def getitem(self):
		# if self.is_pred:
		# 	return self.bc,self.variable,self.coordinate,self.mean,self.std
		# else:
		# train_index = np.random.choice(self.length, batch_size, replace=False)
		train_x = self.x
		train_y = self.y
		train_x = torch.Tensor(train_x).permute(1,0,2,3,4)
		train_y = torch.Tensor(train_y).permute(1,0,2,3,4)

		return train_x,train_y
	def __getitem__(self,index):

		train_x = self.x[:,index]
		train_y = self.y[:,index]
		return train_x,train_y

'''
# import the data
datafile = np.loadtxt(r'F:/1-ML/Fluent_error_compare/empty/empty_files/dp0/FLU-5/Fluent/2D_turbulence')

# split
x = np.round(datafile[:,0],2)
y = np.round(datafile[:,1],2)
p = datafile[:,2]
u = datafile[:,3]
v = datafile[:,4]

# set the index
eps = 1e-6
indices = [[np.where((i/2-eps<=datafile[:,0]) & (datafile[:,0]<=(i+1)/2+eps) & (j/2-eps<=datafile[:,1]) & (datafile[:,1]<=(j+1)/2+eps)) for j in range(6)] for i in range(16)]		# 8*3*10201

# 将数据分为8*3*10201的数组
xx = np.zeros((16,6,2601))
yy = np.zeros((16,6,2601))
pp = np.zeros((16,6,2601))
uu = np.zeros((16,6,2601))
vv = np.zeros((16,6,2601))
for i in range(16):
	for j in range(6):
		if 8<=i<=9 and 0<=j<=1:
			continue
		xx[i,j,:] = x[indices[i][j][0]]
		yy[i,j,:] = y[indices[i][j][0]]
		pp[i,j,:] = p[indices[i][j][0]]
		uu[i,j,:] = u[indices[i][j][0]]
		vv[i,j,:] = v[indices[i][j][0]]
xx = xx.reshape(16,6,51,51)
yy = yy.reshape(16,6,51,51)
pp = pp.reshape(16,6,51,51)
uu = uu.reshape(16,6,51,51)
vv = vv.reshape(16,6,51,51)

# 现在有8*3*51*51的uvp数据，需要对其数据强化
u_original = np.empty((0, 51, 51))
v_original = np.empty((0, 51, 51))
p_original = np.empty((0, 51, 51))

# split = 4
# for i in range(16-1):
# 	for j in range(6-1):
# 		for m in range(split):
# 			for n in range(split):
# 				u_temp0 = uu[i,j,m*51//split:,n*51//split:]
# 				u_temp1 = uu[i+1,j,m*51//split:,:n*51//split]
# 				u_temp2 = uu[i,j+1,:m*51//split,n*51//split:]
# 				u_temp3 = uu[i+1,j+1,:m*51//split,:n*51//split]
# 				u_temp01 = np.concatenate((u_temp0,u_temp1),axis = 1)
# 				u_temp23 = np.concatenate((u_temp2,u_temp3),axis = 1)
# 				u_temp = np.concatenate((u_temp01,u_temp23),axis = 0).reshape(1,51,51)
# 				u_original = np.vstack((u_original,u_temp))

# 				v_temp0 = vv[i,j,m*51//split:,n*51//split:]
# 				v_temp1 = vv[i+1,j,m*51//split:,:n*51//split]
# 				v_temp2 = vv[i,j+1,:m*51//split,n*51//split:]
# 				v_temp3 = vv[i+1,j+1,:m*51//split,:n*51//split]
# 				v_temp01 = np.concatenate((v_temp0,v_temp1),axis = 1)
# 				v_temp23 = np.concatenate((v_temp2,v_temp3),axis = 1)
# 				v_temp = np.concatenate((v_temp01,v_temp23),axis = 0).reshape(1,51,51)
# 				v_original = np.vstack((v_original,v_temp))

# 				p_temp0 = pp[i,j,m*51//split:,n*51//split:]
# 				p_temp1 = pp[i+1,j,m*51//split:,:n*51//split]
# 				p_temp2 = pp[i,j+1,:m*51//split,n*51//split:]
# 				p_temp3 = pp[i+1,j+1,:m*51//split,:n*51//split]
# 				p_temp01 = np.concatenate((p_temp0,p_temp1),axis = 1)
# 				p_temp23 = np.concatenate((p_temp2,p_temp3),axis = 1)
# 				p_temp = np.concatenate((p_temp01,p_temp23),axis = 0).reshape(1,51,51)
# 				p_original = np.vstack((p_original,p_temp))

for i in range(16):
	for j in range(6):
		u_original = np.vstack((u_original,uu[i,j,:,:].reshape(1,51,51)))
		v_original = np.vstack((v_original,vv[i,j,:,:].reshape(1,51,51)))
		p_original = np.vstack((p_original,pp[i,j,:,:].reshape(1,51,51)))

variabl = np.stack((u_original,v_original,p_original),axis=3)
print(variabl.shape)

X,Y = np.meshgrid(np.arange(0,0.51,0.01),np.arange(0,0.51,0.01))
x = np.tile(X,(len(variabl),1,1))
y = np.tile(Y,(len(variabl),1,1))

coordinate = np.stack((x,y),axis=3)

# output
with h5.File(r'2D_turbulence_pred.h5','w') as f:
	f.create_dataset('Coordinates',data = coordinate)
	f.create_dataset('Variables',data = variabl)
'''
if __name__=="__main__":
	data_dir = r'I:\ML\Package_model\DeepONet\train_data\transtant'
	datafile = r'I:\ML\Package_model\DeepONet\train_data\train_data_interval_10_split_2_FNO.h5'
	dataset = Dataset(data_dir,is_pred=True,split=1,is_save=True)
	x,y = dataset.getitem()
	print(torch.any(torch.isnan(y)))
	print(x.shape)
	print(y.shape)
