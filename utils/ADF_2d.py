import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class ADF:
	'''
	要求输入的点的矩阵为[n,2]的形式
	'''
	def __init__(self, points_list):
		self.x,self.y = sp.symbols('x y')
		self.points_list = points_list
		self.xc = []
		self.yc = []
		self.d = []
		self.f = []
		self.t = []
		self.h = []
	def x_y(self,x,y):
		a=0
		k=2
		f=0
		order = 4
		return x+y-(x**order+y**order+a*f**k)**(1/order)
	def h_h(self,h):
		if len(h)>1:
			h1 = h[-2]
			h2 = h[-1]
			h3 = self.x_y(h1,h2)
			h[-2] = h3
			h.pop()
			return self.h_h(h)
		else:
			return h
	def calculate(self):
		for i in range(len(self.points_list)):
			if i == len(self.points_list)-1:
				x1 = self.points_list[i][0]
				y1 = self.points_list[i][1]
				x2 = self.points_list[0][0]
				y2 = self.points_list[0][1]
				self.xc.append((x1+x2)/2)
				self.yc.append((y1+y2)/2)
				self.d.append(((x1-x2)**2+(y1-y2)**2)**0.5)
				self.f.append(((self.x-x1)*(y2-y1) - (self.y-y1)*(x2-x1))/self.d[-1])
				self.t.append(1/self.d[-1]*((self.d[-1]/2)**2 - (self.x-self.xc[-1])**2 - (self.y-self.yc[-1])**2))
				self.h.append((self.f[-1]**2 + ((self.t[-1]**2+self.f[-1]**4)**0.5-self.t[-1])**2/4)**0.5)
			else:
				x1 = self.points_list[i][0]
				y1 = self.points_list[i][1]
				x2 = self.points_list[i+1][0]
				y2 = self.points_list[i+1][1]
				self.xc.append((x1+x2)/2)
				self.yc.append((y1+y2)/2)
				self.d.append(((x1-x2)**2+(y1-y2)**2)**0.5)
				self.f.append(((self.x-x1)*(y2-y1) - (self.y-y1)*(x2-x1))/self.d[-1])
				self.t.append(1/self.d[-1]*((self.d[-1]/2)**2 - (self.x-self.xc[-1])**2 - (self.y-self.yc[-1])**2))
				self.h.append((self.f[-1]**2 + ((self.t[-1]**2+self.f[-1]**4)**0.5-self.t[-1])**2/4)**0.5)
		self.h = self.h_h(self.h)[0]
		self.H = sp.lambdify((self.x,self.y),self.h)
		return self.H
	
if __name__ == '__main__':
	points_list = [[0,0],[4,0],[4,1],[5,1],[5,0],[8,0],[8,0.5],[8.5,0.5],[8.5,1],[8,1],[8,3],[0,3],[0,2.5],[-0.5,2.5],[-0.5,2],[0,2]]
	adf = ADF(points_list)
	H = adf.calculate()
	X,Y = np.meshgrid(np.linspace(0,8,100),np.linspace(0,3,100))
	H = H(X,Y)
	print(H.shape)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	count = ax.contourf(X,Y,H,10)
	bar = fig.colorbar(count)
	ax.set_aspect('equal')
	plt.show()