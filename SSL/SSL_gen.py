import argparse
import torch
import os, cv2, math, random

def parse_args():
    desc = "UnsuperLearn"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--load_model', default = False, type=bool, help='if load previously trained model')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'kmnist'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=4001, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=96, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
     
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--digdim', default = 1, type=int, help='Diagram Dimension')
    parser.add_argument('--usezero', type=bool, default=False)
    parser.add_argument('--tdr', default = 30.0, type=float, help='Regularizion For TD')
    parser.add_argument('--newsample', nargs = '+', default = [1024], type=int, help='List of number of sample for each dim')
    parser.add_argument('--newweight', nargs = '+', default = [1.0], type=float, help='List of weight for each dim')

    parser.add_argument('--zerosample', default = 1024, type=int, help='sample for zero dim')
    parser.add_argument('--zeroweight', default = 1.0, type=float, help='weight for zero dim')

    parser.add_argument('--pagen', default = 4, type=int, help='Number of sample for each dim')
    parser.add_argument('--width', default = 1.0, type = float, help = 'Rbf_width')
    parser.add_argument('--gpu', default = "0", type=str, help='GPU number')

    return parser.parse_args()


opt = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ..PPM import MMD, compute_ppm, SamplePa

from torch.utils.tensorboard import SummaryWriter
#from ComputeQuality import ComputeQuality
from csv import writer
from Model import gModel, dModel, f2cModel

writer_board = SummaryWriter()

def compute_gradient_penalty2(D, real_samples, fake_samples, yy2, xx1):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
			
	yy1 = D(interpolates)
	
	d_interpolates =  torch.norm(yy1 - yy2, dim = 1, keepdim=True) - torch.norm(yy1 - xx1, dim = 1, keepdim=True) 

	fake = torch.autograd.Variable(torch.FloatTensor(real_samples.shape[0],1).fill_(1.0).to(device), requires_grad=False)
	# Get gradient w.r.t. interpolates
	gradients = torch.autograd.grad(
		outputs = d_interpolates,
		inputs = interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1) 
	gradient_penalty = ((gradients.norm(2, dim = 1 ) - 1) ** 2).mean() 
	return gradient_penalty

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

small_KID = np.inf
small_TDD = np.inf

batch_size = opt.batch_size
epochs = opt.epoch
test_freq = 20
learning_rate = opt.lr

TDR = opt.tdr
order = opt.digdim

output_dim = opt.output_dim

UseZero = opt.usezero
NewSample = opt.newsample
NewWeight = opt.newweight 
ZeroSample = [opt.zerosample]
ZeroWeight = [opt.zeroweight]

PaGen = opt.pagen
width = opt.width

path = opt.dataset + "_TDR_" + str(TDR) + "_Width" + str(width)

if(UseZero):
    for i in range(len(NewSample)):
        ZeroSample.append(NewSample[i])
        ZeroWeight.append(NewWeight[i])
    Sample = ZeroSample
    Weight = ZeroWeight
else:
    Sample = NewSample
    Weight = NewWeight

for i in range(len(Sample)):
    path = path + "_Sample_" + str(Sample[i])
for i in range(len(Weight)):
    path = path + "_weight_" + str(Weight[i])

isExist = os.path.exists(path)
if( not isExist):
    os.makedirs(path)

PaGen = opt.pagen
width = opt.width
load_model = opt.load_model

transform = transforms.Compose([transforms.ToTensor()])

if opt.dataset == 'mnist':
    training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
elif opt.dataset == 'fmnist':
    training_set = datasets.FashionMNIST('data/Fmnist', train=True, download=True, transform=transform)
elif opt.dataset == 'kmnist':
    training_set = datasets.KMNIST('data/Kmnist', train=True, download=True, transform=transform)

full_loader = torch.utils.data.DataLoader(training_set, batch_size = batch_size*2, shuffle = True, drop_last = True, pin_memory = True, num_workers = 16)
TotalSize = len(full_loader.dataset) 

epoch_start = 0

g = gModel()
d = dModel(out_dim = 64)

optimizer_g = torch.optim.Adam(g.parameters(), lr = learning_rate, betas=(0.0,0.99))
optimizer_d = torch.optim.Adam(d.parameters(), lr = learning_rate, betas=(0.5,0.99))

if(load_model):
	print('loading model')
	checkpoint_g = torch.load( path + "/g_dev.pth")
	g.load_state_dict(checkpoint_g['model_state_dict'])
	optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])


	checkpoint_d = torch.load( path + "/d_dev.pth")
	d.load_state_dict(checkpoint_d['model_state_dict'])
	optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
	epoch_start = checkpoint_d['epoch'] + 1

	optimizer_to(optimizer_g, device)
	optimizer_to(optimizer_d, device)

g.to(device)
d.to(device)

mmd = MMD(width = width, weights = Weight, num_samples = Sample)

start = torch.cuda.Event(enable_timing = True)
end = torch.cuda.Event(enable_timing = True)

#print(epochs - epoch_start)

genPa = SamplePa(batch_size*2, order, Sample, PaGen, UseZero, device)

for epoch in range(epochs - epoch_start):
	print("Training epoch: " + str(epoch + epoch_start) )
	for i,  (data, label) in enumerate(full_loader):
		data = data.to(device)
		
		if ( ((epoch + epoch_start)*int(math.floor(TotalSize/(2*batch_size))) + i + 1 )%6 == 0 ):

			l_space_pre = (torch.randn(batch_size*2, output_dim)).to(device)
			fake = g(l_space_pre)
			yy = d(fake)
			yys = torch.split(yy, batch_size, dim = 0)

			with torch.no_grad():
				xx = d(data)
				Dx_indep = compute_ppm(X = xx, max_order = order, num_samples = Sample, precompute = True, pa = genPa, use_zero = UseZero, device = device)	
				
			Dy_indep = compute_ppm(X = yy, max_order = order, num_samples = Sample, precompute = True, pa = genPa, use_zero = UseZero, device = device)	

			TLoss = mmd(Dx_indep, Dy_indep)

			g_lossPre = torch.norm(xx[0:batch_size,:] - yy[0:batch_size,:], dim = 1)  - torch.norm(yy[0:batch_size,:] - yy[batch_size:,:], dim = 1) + torch.norm(xx[0:batch_size,:] - yy[batch_size:,:], dim = 1)
			g_loss = g_lossPre.mean() + TDR*TLoss

			writer_board.add_scalar("GTDLoss/train", TLoss, (epoch)*int(TotalSize/(2*batch_size)) + i )
			writer_board.add_scalar("GLoss/train", g_loss, (epoch)*int(TotalSize/(2*batch_size)) + i )
			optimizer_g.zero_grad(set_to_none=True)
			g_loss.backward()
			optimizer_g.step()

		else:

			with torch.no_grad():
				l_space_pre = (torch.randn(batch_size*2, output_dim)).to(device)
				fake = g(l_space_pre)
				fake1 = torch.split(fake, batch_size, dim = 0)
				

			data1 = torch.split(data, batch_size, dim = 0)
			
			xx = d(data)
			
			yy = d(fake)
			Dy_indep = compute_ppm(X = yy, max_order = order, num_samples = Sample, precompute = True, pa = genPa, use_zero = UseZero, device = device)	
			Dx_indep = compute_ppm(X = xx, max_order = order, num_samples = Sample, precompute = True, pa = genPa, use_zero = UseZero, device = device)	
			
			TLoss = mmd(Dx_indep, Dy_indep)   

			d_loss_pre = (torch.norm(xx[0:batch_size,:] - yy[0:batch_size,:], dim = 1) - torch.norm(xx[0:batch_size,:] - xx[batch_size:,:], dim = 1)) - (torch.norm(yy[0:batch_size,:] - yy[batch_size:,:], dim = 1) - torch.norm(yy[0:batch_size,:] - xx[batch_size:,:], dim = 1))
			d_gd = compute_gradient_penalty2(d, data[0:batch_size,:,:,:], fake[0:batch_size,:,:,:], yy[batch_size:,:], xx[batch_size:,:])

			d_loss = -d_loss_pre.mean() + 10*d_gd - TDR*TLoss

			writer_board.add_scalar("DTDLoss/train", TLoss, (epoch)*int(TotalSize/(2*batch_size)) + i )
			writer_board.add_scalar("DLoss/train", d_loss, (epoch)*int(TotalSize/(2*batch_size)) + i ) 
			optimizer_d.zero_grad(set_to_none=True)
			d_loss.backward()
			optimizer_d.step()

	if (epoch + epoch_start) % test_freq == 0: 
		with torch.no_grad():

			X = g((torch.randn(64, output_dim)).to(device))
			X = X.cpu().numpy()
			img = np.zeros((28*8,28*8) )
			for l1 in range(8):
				for l2 in range(8): 
					img[l1*28:(l1+1)*28, l2*28:(l2+1)*28] = X[ l1*8 + l2,0,:,:]
			
			cv2.imwrite( path + '/' + str(epoch + epoch_start) +'.jpg', img*255)
			
			torch.save({
			'epoch': epoch + epoch_start,
			'model_state_dict': g.state_dict(),
			'optimizer_state_dict': optimizer_g.state_dict(),
			}, path + '/' + "g_dev.pth")

			#torch.save(d.state_dict(), "d_dev.pth")
			
			torch.save({
			'epoch': epoch + epoch_start,
			'model_state_dict': d.state_dict(),
			'optimizer_state_dict': optimizer_d.state_dict(),
			}, path + '/' + "d_dev.pth")

	if (epoch + epoch_start) % 400 == 0:

		torch.save({
		'epoch': epoch + epoch_start,
		'model_state_dict': g.state_dict(),
		'optimizer_state_dict': optimizer_g.state_dict(),
		}, path + '/' + str((epoch + epoch_start)) + "g_dev.pth")

		torch.save({
		'epoch': epoch + epoch_start,
		'model_state_dict': d.state_dict(),
		'optimizer_state_dict': optimizer_d.state_dict(),
		}, path + '/' + str((epoch + epoch_start)) + "d_dev.pth")

writer_board.flush()
	

