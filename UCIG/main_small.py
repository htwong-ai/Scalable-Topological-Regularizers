import argparse
import torch
import sys
import os
from pathlib import Path

current_path = Path.cwd()
parent_directory = current_path.parent
sys.path.append(str(parent_directory))

## INPUT ARGUMENTS #########################################################
parser = argparse.ArgumentParser(description='Train Image Generation Models')

# Parse arguments
parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--load_model', default = False, type=bool, help='if load previously trained model')
parser.add_argument('--data', default = "CelebA", type=str, help='Select Data Set')
parser.add_argument('--data_path', default = "/home/user/Downloads/Img", type=str, help='Select Data Set path')
parser.add_argument('--gpu', default = "0", type=str, help='GPU number')
parser.add_argument('--tdr', default = 1000.0, type=float, help='Regularizion For TD')
parser.add_argument('--order', default = 1, type=int, help='Diagram Dimension')
parser.add_argument('--use_zero', default = False, type=bool, help='Use Zero Diagram Dimension')
parser.add_argument('--sample', nargs = '+', default = [1024], type=int, help='List of number of sample for each dim')
parser.add_argument('--weight', nargs = '+', default = [1.0], type=float, help='List of weight for each dim')
parser.add_argument('--width', default = 0.01, type = float, help = 'Rbf_width')
parser.add_argument('--precompute', default = True, type = bool, help = 'Precompute subsamples')
parser.add_argument('--epoch', type=int, default=4001, help='The number of epochs to run')

# Parse arguments
opt = parser.parse_args()
load_model = opt.load_model
gpu_num = opt.gpu

TDR = opt.tdr
order = opt.order
use_zero = opt.use_zero
num_samples = opt.sample
weights = opt.weight
width = opt.width
precompute = opt.precompute
data_path = opt.data_path

PaGen = 8 # used for generating precomputed subsamples

# Check inputs
total_num_dimensions = order
if use_zero:
    total_num_dimensions += 1

if not len(num_samples) == total_num_dimensions:
    raise ValueError("Inconsistent number of samples specified.")

if not len(weights) == total_num_dimensions:
    raise ValueError("Inconsistent number of weights specified.")

print("Using GPU " + gpu_num)
print("UseZero: " + str(use_zero))

# Set up GPU
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
torch.backends.cudnn.benchmark = True


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import math
from numpy import expand_dims
from model import Generator32, ML32
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ComputeQualityWD import ComputeQuality
from PPM import compute_ppm, MMD, SamplePa
from csv import writer
import time
from scipy.special import binom


def compute_gradient_penalty2(D, real_samples1, fake_samples1, yy2, xx1):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = torch.FloatTensor(np.random.random((real_samples1.size(0), 1, 1, 1))).to(device)
	# Get random interpolation between real and fake samples
	interpolates1 = (alpha * real_samples1 + ((1 - alpha) * fake_samples1)).requires_grad_(True)
	
	yy1 = D(interpolates1)
	d_interpolates1 =  torch.norm(yy1 - yy2, dim = 1, keepdim=True) - torch.norm(yy1 - xx1, dim = 1, keepdim=True)

	fake = torch.autograd.Variable(torch.FloatTensor(real_samples1.shape[0],1).fill_(1.0).to(device), requires_grad=False)
	# Get gradient w.r.t. interpolates
	gradients1 = torch.autograd.grad(
		outputs = d_interpolates1,
		inputs = interpolates1,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients1 = gradients1.view(gradients1.size(0), -1) 
	gradient_penalty = ((gradients1.norm(2, dim = 1 ) - 1) ** 2).mean() 
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

# Training and save parameters
beta1 = 0.0
beta2 = 0.99
batch_size = 96
out_dim = 128
learning_rate = 1e-4
Size = 32

epochs = opt.epoch
epoch_start = 0
save_epoch = -1

# Save parameters
test_freq = 20 # save frequency
QCheck_freq = 160
    
small_CMMD = np.inf
small_WD = np.inf

transform = transforms.Compose([
        transforms.Resize(Size),
        transforms.CenterCrop(Size),
        transforms.ToTensor()
    ])

# Set output path
#path = output_path + "TDR_" + str(TDR) + "_Order_" + str(order) + "_Width" + str(width)

path = opt.data + "_TDR_" + str(TDR) + "_Order_" + str(order) + "_Width" + str(width)

if(use_zero):
    path = path + "_usezero"

for i in range(len(num_samples)):
    path = path + "_Sample_" + str(num_samples[i])
for i in range(len(weights)):
    path = path + "_weight_" + str(weights[i])

isExist = os.path.exists(path)
if( not isExist):
    os.makedirs(path)

FileDir = path + '/metric.csv'
writer_board = SummaryWriter(log_dir = path+"/"+datetime.now().strftime("%Y%m%d%H%M%S%z"))

# Load dataset
if(opt.data == "CelebA"):
    CelebA = datasets.CelebA(root = data_path, split = "train", transform = transform, download=True)
    train_loader = torch.utils.data.DataLoader(CelebA, batch_size = batch_size*2, shuffle = True, drop_last = True, pin_memory = True, num_workers = 16)
elif(opt.data == "AnimeFace"):
    AnimeFace = datasets.ImageFolder(root = data_path, transform = transform)
    train_loader = torch.utils.data.DataLoader(AnimeFace, batch_size = batch_size*2, shuffle = True, drop_last = True, pin_memory = True, num_workers = 16)

TotalSize = len(train_loader.dataset) 

netG = Generator32()  
netML = ML32(out_dim=out_dim)

optimizerG = torch.optim.Adam(netG.parameters(), lr = learning_rate, betas=(beta1,beta2)) 
optimizerML = torch.optim.Adam(netML.parameters(), lr = learning_rate, betas=(0.5,0.99))

if(load_model):
    print('loading model')

    G_load = torch.load(path + '/' + "g.pth")
    ML_load = torch.load(path + '/' + "d.pth")

    netG.load_state_dict(G_load["G_State_Dict"])
    netML.load_state_dict(ML_load["ML_State_Dict"])

    optimizerG.load_state_dict(G_load["OptimizerG_State_Dict"])
    optimizerML.load_state_dict(ML_load["OptimizerML_State_Dict"])

    epoch_start = ML_load['epoch'] + 1

    optimizer_to(optimizerG, device)
    optimizer_to(optimizerML, device)

netG = netG.to(device)
netML = netML.to(device)

# Initialize MMD
mmd = MMD(width = width, weights = weights, num_samples = num_samples, device = device)

# Generated precomputed subsamples if required
pa = None
if precompute:
    pa = SamplePa(batch_size*2, order, num_samples, PaGen, use_zero, device)

for epoch in range(epochs - epoch_start): 
    print("Epoch: " + str(epoch + epoch_start), flush=True)
    for i,  (data, label) in enumerate(train_loader):
        timer_start = time.time()
        data = data.to(device)
        #TLoss = 0.0
        # Train generator
        if ( (epoch*int(math.floor(TotalSize/(2*batch_size))) + i + 1 )%6 == 0 ):

            l_space_pre = (torch.rand(batch_size*2, out_dim)*2-1).to(device)
            fake = netG(l_space_pre)
            
            yy = netML(fake)
            yys = torch.split(yy, batch_size, dim = 0)
            
            with torch.no_grad():
                xx = netML(data)
                xxs = torch.split(xx, batch_size, dim = 0)
                Dx_indep = compute_ppm(xx, order, num_samples, precompute, pa, use_zero, device)
            
            Dy_indep = compute_ppm(yy, order, num_samples, precompute, pa, use_zero, device)
            TLoss = mmd(Dy_indep, Dx_indep)

            g_lossPre = torch.norm(xxs[0] - yys[0], dim = 1)  - torch.norm(yys[0] - yys[1], dim = 1) + torch.norm(xxs[0] - yys[1], dim = 1)
            g_loss = g_lossPre.mean() + TDR*TLoss

            writer_board.add_scalar("GTLoss/train", TLoss, epoch*int( math.floor(TotalSize/(2*batch_size)) ) + i )
            writer_board.add_scalar("GLoss/train", g_loss, epoch*int( math.floor(TotalSize/(2*batch_size)) ) + i )
            optimizerG.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizerG.step()

        else:

            with torch.no_grad():
                l_space_pre = (torch.rand(batch_size*2, out_dim)*2-1).to(device)
                fake = netG(l_space_pre)
                fake1 = torch.split(fake, batch_size, dim = 0)
            
            data1 = torch.split(data, batch_size, dim = 0)
            xx = netML(data)
            yy = netML(fake)

            xxs = torch.split(xx, batch_size, dim = 0)
            yys = torch.split(yy, batch_size, dim = 0)
            
            Dx_indep = compute_ppm(xx, order, num_samples, precompute, pa, use_zero, device)
            Dy_indep = compute_ppm(yy, order, num_samples, precompute, pa, use_zero, device)

            TLoss = mmd(Dy_indep, Dx_indep)
                
            d_loss_pre = (torch.norm(xxs[0] - yys[1], dim = 1) - torch.norm(xxs[0] - xxs[1], dim = 1)) - (torch.norm(yys[0] - yys[1], dim = 1) - torch.norm(yys[0] - xxs[1], dim = 1))
            d_gd = compute_gradient_penalty2(netML, data1[0], fake1[0], yys[1], xxs[1])
            d_loss = -d_loss_pre.mean() + 10*d_gd - TDR*TLoss

            writer_board.add_scalar("DLoss/train", d_loss, epoch*int( math.floor(TotalSize/(2*batch_size)) ) + i )
            writer_board.add_scalar("DTLoss/train", TLoss, epoch*int( math.floor(TotalSize/(2*batch_size)) )  + i ) 
            optimizerML.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizerML.step()
            
    
    if (epoch + epoch_start) % QCheck_freq == 0:

        with torch.no_grad():

            if(opt.data == "CelebA"):
                CMMD, WD = ComputeQuality(netG, 32, 100, "cuda:0", CelebA, out_dim)
            elif(opt.data == "AnimeFace"):
                CMMD, WD = ComputeQuality(netG, 32, 100, "cuda:0", AnimeFace, out_dim)
            
        
        print("Epoch: " + str(epoch + epoch_start))
        print("CMMD: " + str(CMMD))
        print("WD: " + str(WD))


        DataList = [ epoch + epoch_start, CMMD.cpu().numpy(), WD.cpu().numpy() ]

        with open(FileDir, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(DataList)
            f_object.close()

        if( (epoch + epoch_start) > save_epoch ):

            torch.save({
            'epoch': epoch + epoch_start,
            'G_State_Dict': netG.state_dict(),
            'OptimizerG_State_Dict': optimizerG.state_dict(),
            }, path + '/' + str(epoch + epoch_start) + "g.pth")

            #torch.save(d.state_dict(), "d_dev.pth")

            torch.save({
            'epoch': epoch + epoch_start,
            'ML_State_Dict': netML.state_dict(),
            'OptimizerML_State_Dict': optimizerML.state_dict(),
            }, path + '/' + str(epoch + epoch_start) + "d.pth")


        if(CMMD < small_CMMD):
            print("small CMMD updated")

            torch.save({
            'epoch': epoch + epoch_start,
            'G_State_Dict': netG.state_dict(),
            'OptimizerG_State_Dict': optimizerG.state_dict(),
            }, path + '/' + "g_Small_CMMD.pth")

            torch.save({
            'epoch': epoch + epoch_start,
            'ML_State_Dict': netML.state_dict(),
            'OptimizerML_State_Dict': optimizerML.state_dict(),
            }, path + '/' + "d_Small_CMMD.pth")

            small_CMMD = CMMD

        if(WD < small_WD):
            print("small WD updated")

            torch.save({
            'epoch': epoch + epoch_start,
            'G_State_Dict': netG.state_dict(),
            'OptimizerG_State_Dict': optimizerG.state_dict(),
            }, path + '/' +"g_Small_WD.pth")

            torch.save({
            'epoch': epoch + epoch_start,
            'ML_State_Dict': netML.state_dict(),
            'OptimizerML_State_Dict': optimizerML.state_dict(),
            }, path + '/' + "d_Small_WD.pth")

            small_WD = WD


    if (epoch + epoch_start) % test_freq == 0: 
        with torch.no_grad():

            X = netG((torch.rand(36, out_dim)*2-1).to(device))
            X = X.cpu().numpy()
            img = np.zeros((3, Size*6, Size*6) )
            for l1 in range(6):
                for l2 in range(6): 
                    img[2, l1*Size:(l1+1)*Size, l2*Size:(l2+1)*Size] = X[ l1*6 + l2, 0,:,:]
                    img[1, l1*Size:(l1+1)*Size, l2*Size:(l2+1)*Size] = X[ l1*6 + l2, 1,:,:]
                    img[0, l1*Size:(l1+1)*Size, l2*Size:(l2+1)*Size] = X[ l1*6 + l2, 2,:,:]
            img = np.transpose(img,(1,2,0))
            cv2.imwrite( path + '/' + str(epoch + epoch_start) + 'ss.jpg', img*255)
            torch.save({'epoch': epoch + epoch_start, 'G_State_Dict': netG.state_dict(), 'OptimizerG_State_Dict': optimizerG.state_dict()}, path + '/' + "g.pth")
            torch.save({'epoch': epoch + epoch_start, 'ML_State_Dict': netML.state_dict(), 'OptimizerML_State_Dict': optimizerML.state_dict()}, path + '/' + "d.pth")
    
writer_board.flush()
        

