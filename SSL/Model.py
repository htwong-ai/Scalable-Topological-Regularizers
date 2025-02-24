import torch

class gModel(torch.nn.Module):
	def __init__(self):
		super(gModel, self).__init__()
		self.linear1 = torch.nn.Linear(64, 128, bias = False)
		self.linear2 = torch.nn.Linear(128, 64*3*3, bias = False)
		self.cov2dt1 = torch.nn.ConvTranspose2d(64, 64, (3,3), stride = (1,1), bias = False)
		self.cov2dt2 = torch.nn.ConvTranspose2d(64, 32, (3,3), stride = (1,1))
		self.cov2dt3 = torch.nn.ConvTranspose2d(32, 32, (5,5), stride = (1,1), bias = False)
		self.cov2dt4 = torch.nn.ConvTranspose2d(32, 1, (5,5), stride = (1,1))
		

	def forward(self, x):
		x = self.linear1(x).clamp(min = 0)
		x = self.linear2(x)
		#x = self.linear3(x)
		x = torch.reshape(x, (x.size()[0],64, 3,3))
		x = torch.nn.functional.interpolate(x, size = (6,6), mode = 'area')
		x = self.cov2dt1(x).clamp(min = 0)
		x = self.cov2dt2(x).clamp(min = 0)
		x = torch.nn.functional.interpolate(x, size = (20,20), mode = 'area')
		x = self.cov2dt3(x).clamp(min = 0)
		x = self.cov2dt4(x)  	
		x = torch.sigmoid(x)

		return x

class dModel(torch.nn.Module):
	def __init__(self, out_dim = 84):
		super(dModel, self).__init__()
		self.conv2d1 = torch.nn.Conv2d(1, 32,(5,5), stride = (1,1))
		self.conv2d2 = torch.nn.Conv2d(32, 32,(5,5), stride = (1,1), bias = False)
		self.conv2d3 = torch.nn.Conv2d(32, 64,(3,3), stride = (1,1))
		self.conv2d4 = torch.nn.Conv2d(64, 64,(3,3), stride = (1,1), bias = False)
		self.avgpool = torch.nn.AvgPool2d(2, stride = 2)
		self.linear1 = torch.nn.Linear(64*3*3, 128, bias = False)
		self.linear2 = torch.nn.Linear(128, out_dim ,bias = False)
		
		self.LeakyReLu = torch.nn.LeakyReLU(0.1)

	def forward(self, x):

		fe1 = self.LeakyReLu(self.conv2d1(x))
		
		fe2 = self.LeakyReLu(self.conv2d2(fe1))
		fe2 = self.avgpool(fe2)
		
		fe3 = self.LeakyReLu(self.conv2d3(fe2))
		
		fe4 = self.LeakyReLu(self.conv2d4(fe3))	
		fe4 = self.avgpool(fe4)			
		
		fe4 = torch.flatten(fe4, start_dim = 1)
		
		out = self.LeakyReLu(self.linear1(fe4))
		out = self.linear2(out)

		return out

class f2cModel(torch.nn.Module):
	def __init__(self, encode_dim):
		super(f2cModel, self).__init__()
		
		self.input_l1 = torch.nn.Linear(64, encode_dim)

		self.linear1 = torch.nn.Linear(1*encode_dim, 1*encode_dim, bias = False)
		self.linear2 = torch.nn.Linear(1*encode_dim, encode_dim, bias = False)
		self.linear3 = torch.nn.Linear(encode_dim, 10)

		self.LeakyReLu = torch.nn.LeakyReLU(0.1)
		#self.softmax = torch.nn.Softmax()

	def forward(self, x1):

		f1 = self.input_l1(x1)

		x = self.LeakyReLu(f1)
		x = self.LeakyReLu(self.linear1(x))
		x = self.LeakyReLu(self.linear2(x))
		out = self.linear3(x)

		return torch.nn.functional.log_softmax(out, dim=1)

class f2cModel_draw(torch.nn.Module):
	def __init__(self, encode_dim):
		super(f2cModel_draw, self).__init__()

		self.input_l1 = torch.nn.Linear(64, encode_dim)

		self.linear1 = torch.nn.Linear(1*encode_dim, 1*encode_dim, bias = False)
		self.linear2 = torch.nn.Linear(1*encode_dim, encode_dim, bias = False)
		self.linear3 = torch.nn.Linear(encode_dim, 10)

		self.LeakyReLu = torch.nn.LeakyReLU(0.1)
		self.softmax = torch.nn.Softmax()

	def forward(self, x1):

		f1 = self.input_l1(x1)

		x = self.LeakyReLu(f1)
		x = self.LeakyReLu(self.linear1(x))
		x = self.LeakyReLu(self.linear2(x))
		out = self.linear3(x)

		return torch.nn.functional.log_softmax(out, dim=1)



