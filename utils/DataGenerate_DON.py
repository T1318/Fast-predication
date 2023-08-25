import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import torch
from ADF_2d import ADF
'''
fluent导出文件为: x y p u v
经过处理后文件为: x y u v p
'''
class Dataset_DON():
	def __init__(self,datafile_name,points_list,ratio,is_pred=False):
		self.ratio = ratio
		self.points_list = points_list
		self.datafile = np.loadtxt(datafile_name)
		self.is_pred = is_pred
		self.variable, self.coordinate = self.enhance_data()
		self.length = self.variable.shape[0]
		bc_left = self.variable[:, :, 0, :]
		bc_right = self.variable[:, :, -1, :]
		bc_top = self.variable[:, -1, :, :]
		bc_bottom = self.variable[:, 0, :, :]
		# 找到最大进风的方向,result表示[left,top,right,bottom]哪个方向的进风最大,表示索引。self.bc表示最大进风的边界条件[batch_size,51,3]
		# v_mean_left = np.mean(bc_left,axis=1)[:,0]
		# v_mean_right = -np.mean(bc_right,axis=1)[:,0]
		# v_mean_top = -np.mean(bc_top,axis=1)[:,1]
		# v_mean_bottom = np.mean(bc_bottom,axis=1)[:,1]
		# v_mean = np.stack((v_mean_left,v_mean_top,v_mean_right,v_mean_bottom),axis=1)
		# max_index = np.argmax(v_mean, axis=1)
		# self.result = np.eye(4)[max_index]
		# self.bc = bc_left * self.result[:,None,None,0] + bc_top * self.result[:,None,None,1] + bc_right * self.result[:,None,None,2] + bc_bottom * self.result[:,None,None,3]
		self.bc = np.concatenate((bc_left, bc_top, bc_right, bc_bottom), axis=1)
		self.min = np.zeros((self.bc.shape[0],self.bc.shape[-1]))
		self.max = np.zeros((self.bc.shape[0],self.bc.shape[-1]))
		# normalize
		self.bc,self.variable = self.normalize(self.bc,self.variable)
		# reshape
		self.coordinate = self.coordinate.reshape(self.length,-1,2)
		self.variable = self.variable.reshape(self.length,-1,3)
	def __len__(self):
		return self.length
	def enhance_data(self):
		# split
		x = np.round(self.datafile[:,0],2)
		y = np.round(self.datafile[:,1],2)
		p = self.datafile[:,2]
		u = self.datafile[:,3]
		v = self.datafile[:,4]

		# set the index
		eps = 1e-6
		indices = [[np.where((i/2-eps<=self.datafile[:,0]) & (self.datafile[:,0]<=(i+1)/2+eps) & (j/2-eps<=self.datafile[:,1]) & (self.datafile[:,1]<=(j+1)/2+eps)) for j in range(6)] for i in range(16)]		# 8*3*10201
		# 将数据分为8*3*10201的数组
		xx = np.zeros((16,6,2601))
		yy = np.zeros((16,6,2601))
		pp = np.zeros((16,6,2601))
		uu = np.zeros((16,6,2601))
		vv = np.zeros((16,6,2601))
		for i in range(16):
			for j in range(6):
				# if 4<=i<=5 and 0<=j<=1:
				# 	continue
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
		if self.is_pred:
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
			split = 4
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

		variables = np.stack((u_original,v_original,p_original),axis=3)

		X,Y = np.meshgrid(np.arange(0,0.51,0.01),np.arange(0,0.51,0.01))
		x = np.tile(X,(len(variables),1,1))
		y = np.tile(Y,(len(variables),1,1))

		coordinates = np.stack((x,y),axis=3)
		# coordinates = np.stack((x_original,y_original),axis=3)
		'''
		variables:		[batch_size,51,51,3]
		coordinates:	[batch_size,51,51,2]
		distance:		[batch_size,51,51,1]
		'''
		return variables,coordinates
	def normalize(self,bc,variable):
		'''
		min:	[batch_size,2],		0:u,v,	1:p
		max:	[batch_size,2],		0:u,v,	1:p
		'''
		for i in range(bc.shape[0]):
			for j in range(bc.shape[-1]):
				self.min[i,j] = np.min(bc[i,:,j])
				self.max[i,j] = np.max(bc[i,:,j])
				bc[i,:,j] = (bc[i,:,j]-self.min[i,j])/(self.max[i,j]-self.min[i,j]+1e-6)
				variable[i,:,:,j] = (variable[i,:,:,j]-self.min[i,j])/(self.max[i,j]-self.min[i,j]+1e-6)
		return bc,variable
	def inverse_normalize(self,variable,min,max):
		for i in range(variable.shape[0]):
			for j in range(variable.shape[-1]):
				variable[i,:,:,j] = variable[i,:,:,j]*(max[i,j]-min[i,j]+1e-6)+min[i,j]
		return variable
	def get_data(self):
		if self.is_pred:
			return self.bc,self.variable,self.coordinate[0],self.min,self.max
		else:
			train_index = np.random.choice(self.length,int(self.length*self.ratio),replace=False)
			test_index = np.setdiff1d(np.arange(self.length),train_index,assume_unique=True)
			train_bc = self.bc[train_index]
			train_variable = self.variable[train_index]
			train_coordinate = self.coordinate[0]
			train_min = self.min[train_index]
			train_max = self.max[train_index]
			test_bc = self.bc[test_index]
			test_variable = self.variable[test_index]
			test_coordinate = self.coordinate[0]
			test_min = self.min[test_index]
			test_max = self.max[test_index]
			return train_bc,train_variable,train_coordinate,train_min,train_max,test_bc,test_variable,test_coordinate,test_min,test_max





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