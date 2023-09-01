import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.ADF_2d import ADF
'''
fluent导出文件为: x y p u v
经过处理后文件为: x y u v p
'''
class Dataset_DON():
	def __init__(self,path_label,n_x=16,n_y=6,p_x=51,p_y=51):
		file_list = os.listdir(path_label)
		self.train_label = torch.empty((0,p_x,p_y,1))

		for file in file_list:
			if file.endswith('.npy'):
				label = np.load(os.path.join(path_label, file))
				self.train_label = torch.cat((self.train_label, torch.Tensor(label)), 0)
		
		self.train_label = self.train_label[:, :, :, 0]
		X,Y = np.meshgrid(np.arange(0,0.51,0.01),np.arange(0,0.51,0.01))

		self.coordinate = np.stack((X,Y),axis=2)

		self.length = self.train_label.shape[0]
		bc_left = self.train_label[:, :, 0]
		bc_right = self.train_label[:, :, -1]
		bc_top = self.train_label[:, -1, :]
		bc_bottom = self.train_label[:, 0, :]
		self.bc = np.concatenate((bc_left, bc_top, bc_right, bc_bottom), axis=1)
		self.bc = torch.Tensor(self.bc)
		self.min = torch.zeros(self.bc.shape[0])
		self.max = torch.zeros(self.bc.shape[0])

		# normalize
		self.bc,self.train_label = self.normalize(self.bc,self.train_label)
		# reshape
		self.train_label = self.train_label.reshape(self.length,-1)
	def __len__(self):
		return self.length

	def normalize(self,bc,variable):
		'''
		min:	[batch_size,2],		0:u,v,	1:p
		max:	[batch_size,2],		0:u,v,	1:p
		'''
		for i in range(bc.shape[0]):
				self.min[i] = torch.min(bc[i])
				self.max[i] = torch.max(bc[i])
				bc[i] = (bc[i]-self.min[i])/(self.max[i]-self.min[i]+1e-6)
				variable[i] = (variable[i]-self.min[i])/(self.max[i]-self.min[i]+1e-6)
		return bc,variable
	def inverse_normalize(self,variable,min,max):
		for i in range(variable.shape[0]):
				variable[i] = variable[i]*(max[i]-min[i]+1e-6)+min[i]
		return variable
	def get_data(self):
		self.coordinate = torch.Tensor(self.coordinate).reshape(-1,2)
		train_index = np.random.choice(self.length,int(self.length),replace=False)
		self.bc = self.bc[train_index]
		self.train_label = self.train_label[train_index]
		self.min = self.min[train_index]
		self.max = self.max[train_index]
		return self.bc,self.train_label,self.coordinate,self.min,self.max





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

variables = np.stack((u_original,v_original,p_original),axis=3)
print(variables.shape)

X,Y = np.meshgrid(np.arange(0,0.51,0.01),np.arange(0,0.51,0.01))
x = np.tile(X,(len(variables),1,1))
y = np.tile(Y,(len(variables),1,1))

coordinate = np.stack((x,y),axis=3)

# output
with h5.File(r'2D_turbulence_pred.h5','w') as f:
	f.create_dataset('Coordinates',data = coordinate)
	f.create_dataset('Variables',data = variables)
'''
if __name__=="__main__":
	points_list = [[0,0],[4,0],[4,1],[5,1],[5,0],[8,0],[8,3],[0,3]]
	dataset = Dataset(r'F:/1-ML/Fluent_error_compare/empty/empty_files/dp0/FLU-5/Fluent/2D_turbulence',points_list,1,True)
	bc,variable,coordinate,distance,min,max = dataset.get_data()
	print(bc.shape)
	print(variable.shape)
	print(coordinate.shape)
	print(distance.shape)
	print(min.shape)
	print(max.shape)
	variable = variable.reshape(-1,51,51,3)
	variable = dataset.inverse_normalize(variable,min,max)
	print(variable.shape)
	# print()
	# np.savetxt('bc.txt',bc[0],fmt='%f')