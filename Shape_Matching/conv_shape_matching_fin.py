import torch
import ot
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
from pathlib import Path
# sys.path.append('/content/drive/MyDrive/Colab Notebooks/')

from scipy.stats import bernoulli
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn import WassersteinDistance

from scipy.stats import wasserstein_distance_nd

current_path = Path.cwd()
parent_directory = current_path.parent
sys.path.append(str(parent_directory))

from PPM import MMD, compute_ppm, SamplePa, MMD_RBF_main

from generate_shape import generate_shapes
from csv import writer

def train_loop( pa, ref_shape, num_points, dim_data, learning_rate, momentum, epoch, use_zero, max_order, loss_num, mmd, MMD_RBF_main, MMDweight, re_save_path , device):

	VR = VietorisRipsComplex(dim = 1)
	WD = WassersteinDistance(p = 2)

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	train_shape = torch.nn.parameter.Parameter( torch.randn(num_points,dim_data).to(device)*0.3 , requires_grad=True)
	optimizer = torch.optim.SGD([train_shape], lr=learning_rate,momentum=momentum)
	half = int(num_points/2)

	result_PD = np.zeros(int(epoch/20))
	result_WD = np.zeros(int(epoch/20))

	ep_count = np.zeros(int(epoch/20))

	save_num = 0
	for j in range(epoch):

		total_loss = 0.0

		if loss_num < 4:
			# Cramer loss
			ref_perm = ref_shape[torch.randperm(num_points), :]
			train_perm = train_shape[torch.randperm(num_points), :]
			xxs = torch.split(ref_perm, half, dim=0)
			yys = torch.split(train_perm, half, dim=0)
			C_loss = torch.norm(xxs[0] - yys[0], dim = 1)  - torch.norm(yys[0] - yys[1], dim = 1) + torch.norm(xxs[0] - yys[1], dim = 1)
			main_loss = C_loss.mean()
			total_loss += CR*C_loss.mean()

		elif loss_num > 3:
			main_loss = MMDweight*MMD_RBF_main(ref_shape, train_shape, width_main_cost)
			total_loss += main_loss

		if loss_num == 2 or loss_num == 5:
			# Topological loss
			D_ref = compute_ppm(ref_shape, max_order, num_samples, precompute = False,  pa = pa, use_zero = use_zero, device=device)
			D_train = compute_ppm(train_shape, max_order, num_samples, precompute = False, pa = pa, use_zero = use_zero, device=device)
			TD_loss = mmd(D_ref, D_train)
			total_loss += TDR*TD_loss

		if loss_num == 3:
			# True Distance
			train_shape_Matrix = torch.cdist(train_shape, train_shape)
			VRcomplexT = VR(train_shape_Matrix, treat_as_distances = True)

			ref_shape_Matrix = torch.cdist(ref_shape, ref_shape)
			VRcomplexR = VR(ref_shape_Matrix, treat_as_distances = True)

			TTD_loss = WD(VRcomplexT[1].diagram, VRcomplexR[1].diagram)

			total_loss += TDR*TTD_loss  

		#if(False):
		#	total_loss = total_loss + torch.norm(total_loss - 0.05)

		if(j%20 == 0):
			train_shape_Matrix = torch.cdist(train_shape, train_shape)
			VRcomplexT = VR(train_shape_Matrix, treat_as_distances = True)

			ref_shape_Matrix = torch.cdist(ref_shape, ref_shape)
			VRcomplexR = VR(ref_shape_Matrix, treat_as_distances = True)

			track_conv = WD(VRcomplexT, VRcomplexR)

			M = torch.cdist(ref_shape, train_shape)
			a = (torch.ones(ref_shape.size()[0])/ref_shape.size()[0]).to(device)
			b = (torch.ones(ref_shape.size()[0])/ref_shape.size()[0]).to(device)

			T = ot.emd(a, b, M)

			WD_value = torch.sqrt((T*M).sum())

			result_PD[save_num] = track_conv

			result_WD[save_num] = WD_value

			ep_count[save_num] = j

			PrintCopy2 = ref_shape.cpu().detach().numpy()
			PrintCopy = train_shape.cpu().detach().numpy()
			if dim_data == 2:
				fig = plt.figure(figsize=plt.figaspect(1.0))
				ax = fig.add_subplot()
				ax.scatter(PrintCopy2[:,0], PrintCopy2[:,1], marker='o', s=5)
				ax.scatter(PrintCopy[:,0], PrintCopy[:,1], marker='^', s=5)
				ax.set_xlim([-2.0,2.0])
				ax.set_ylim([-2.0,2.0])
				plt.gca().set_aspect('equal')
				ax.title.set_text('Point Cloud')
			else:
				# In the sphere case, also plot histogram of norms to check that the sphere is actually hollow
				fig = plt.figure(figsize=plt.figaspect(0.333))
				ax = fig.add_subplot(1,2,1,projection='3d')
				ax.scatter(PrintCopy2[:,0], PrintCopy2[:,1], PrintCopy2[:,2], marker='o', s=5)
				ax.scatter(PrintCopy[:,0], PrintCopy[:,1], PrintCopy[:,2],  marker='^', s=5)
				ax.set_xlim([-1.5,1.5])
				ax.set_ylim([-1.5,1.5])
				ax.set_zlim([-1.5,1.5])
				plt.gca().set_aspect('equal')
				ax.title.set_text('Point Cloud')

				ax = fig.add_subplot(1,3,2)
				PrintCopy = train_shape.cpu().detach()
				plt.hist(torch.norm(PrintCopy,dim=1), range=(0,1.5), bins=25)

				ax = fig.add_subplot(1,3,3)

			plt.savefig(re_save_path + str(save_num))
			plt.close()
			save_num += 1

		# Compute total loss and optimize
		# total_loss = CR*C_loss.mean()
		# total_loss = CR*C_loss.mean() + TDR*TD_loss
		optimizer.zero_grad(set_to_none=True)
		total_loss.backward()
		optimizer.step()



	return ep_count, result_PD, result_WD

parser = argparse.ArgumentParser(description='Compute Metric')
parser.add_argument('--shape_num', default = 1, type=int, help='Maximum Epoch')
parser.add_argument('--loss_num', default = 1, type=int, help='Maximum Epoch')
parser.add_argument('--tdr', default = 1.0, type=float, help='Regularizion For TD')
parser.add_argument('--width', default = 0.1, type = float, help = 'Rbf_width')
parser.add_argument('--n_sample', default = 2000, type = int, help = 'Rbf_width')
parser.add_argument('--num_points', default = 512, type = int, help = 'Rbf_width')
parser.add_argument('--sim_n_conv', default = False, type = bool, help = 'Rbf_width')


opt = parser.parse_args()

use_GPU = True

sim_n_conv = opt.sim_n_conv

if use_GPU:
    # Set up GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'


# Reference shape number
# Shape 1: circle
# Shape 2: two intersected circles
shape_num = opt.shape_num


# Loss
# Loss 1: Cramer Vs Cramer + PPM
# Loss 2: MMD Vs_MMD + PPM
loss_num = opt.loss_num

# Training parameters
learning_rate = 5e-2
momentum = 0.9
epoch = 20*121

# Weights for different loss functions 
# Main loss
#CR = 1.0
CR = 1.6
MMDweight = 5.0 # Scale MMD weight so that the scale is roughly the same as cramer

# Regularizer
TDR = opt.tdr

# Precompute subsamples
precompute = False

# Diagram parameters
max_order = 1
use_zero = True

# MMD Parameters
num_samples = [opt.n_sample, opt.n_sample]
weight = [1.0, 6000.0]
decay_exponent = 1.0
width = opt.width
width_main_cost = 0.1 # Width for RBF MMD in main loss function

pa = None
if precompute:
    pa = SamplePa(opt.num_points, max_order, num_samples, 8, use_zero, device)


# Initialize MMD
mmd = MMD(width = width, weights = weight, num_samples = num_samples, decay_exponent = decay_exponent, device = device)
mmd_main_cost = MMD(width = width_main_cost)

# Generate shapes
num_points = opt.num_points
half = int(num_points/2)

# Build reference shape
ref_shape = generate_shapes(shape_num, num_points, device)

# Build training shape
dim_data = 2
if shape_num == 4:
    dim_data = 3
train_shape = torch.nn.parameter.Parameter( torch.randn(num_points,dim_data).to(device)*0.3 , requires_grad=True)

# Generate savepath
save_path = "figures8/"

if shape_num == 1:
    save_path += "circle/"
elif shape_num == 2:
    save_path += "intcircles/"

if loss_num == 1:
    save_path += "cr_Vs_cr_ppm/"
elif loss_num == 2:
    save_path += "mmd_Vs_mmd_ppm/"
save_path = save_path + "LR" + str(learning_rate) + "_M" + str(momentum) + "_O" + str(max_order) + "_TDR" + str(TDR) + "_W" + str(weight[1]) + "_width" + str(width) + "_NS" + str(num_samples[0]) + "/"

# Make folder
print(save_path)
isExist = os.path.exists(save_path)
if( not isExist):
    os.makedirs(save_path)

# Plot reference shape
fig = plt.figure()
if dim_data == 2:
    ax = fig.add_subplot()
    p_ref_shape = ref_shape.detach().cpu().numpy() 
    ax.scatter(p_ref_shape[:,0], p_ref_shape[:,1], marker='o')
    print_train_shape = train_shape.detach().cpu().numpy()
    ax.scatter(print_train_shape[:,0], print_train_shape[:,1], marker='^')
else:    
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ref_shape[:,0], ref_shape[:,1], ref_shape[:,2], marker='o')
    print_train_shape = train_shape.detach().numpy()
    ax.scatter(print_train_shape[:,0], print_train_shape[:,1], print_train_shape[:,2], marker='^')

fig.savefig(save_path+"ref.png")
plt.close(fig)

if loss_num == 1:

	re_save_path = save_path + "nsample_" + str(opt.n_sample) + "_num_points_" +str(opt.num_points) + "_cr/"

	print(re_save_path)
	isExist = os.path.exists(re_save_path)
	if( not isExist):
		os.makedirs(re_save_path)

	e1, p1, w1 = train_loop( pa, ref_shape, num_points, dim_data, learning_rate, momentum, epoch, use_zero, max_order, 1, mmd, MMD_RBF_main, MMDweight, re_save_path, device)
	
	re_save_path = save_path + "nsample_" + str(opt.n_sample) + "_num_points_" +str(opt.num_points) + "_cr+ppm/"
	
	print(re_save_path)
	isExist = os.path.exists(re_save_path)
	if( not isExist):
		os.makedirs(re_save_path)

	e2, p2, w2 = train_loop( pa, ref_shape, num_points, dim_data, learning_rate, momentum, epoch, use_zero, max_order, 2, mmd, MMD_RBF_main, MMDweight, re_save_path, device)

	plt.plot(e1, p1, 'r--', e2, p2, 'b--')
	plt.xlabel('Step', fontsize=12)
	plt.ylabel('$PD_{dist}$', fontsize=12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(["Cramer", "Cramer + PPM-Reg"], loc="upper right", fontsize=12)
	plt.xlim([0, 20*120])
	plt.savefig(save_path + 'PD.png', bbox_inches='tight')
	plt.close()

elif loss_num ==2 :

	re_save_path = save_path + "nsample_" + str(opt.n_sample) + "_num_points_" +str(opt.num_points) + "_mmd/"

	print(re_save_path)
	isExist = os.path.exists(re_save_path)
	if( not isExist):
		os.makedirs(re_save_path)

	e1, p1, w1 = train_loop( pa, ref_shape, num_points, dim_data, learning_rate, momentum, epoch, use_zero, max_order, 4, mmd, MMD_RBF_main, MMDweight, re_save_path, device)

	re_save_path = save_path + "nsample_" + str(opt.n_sample) + "_num_points_" +str(opt.num_points) + "_mmd+ppm/"

	print(re_save_path)
	isExist = os.path.exists(re_save_path)
	if( not isExist):
		os.makedirs(re_save_path)

	e2, p2, w2 = train_loop( pa, ref_shape, num_points, dim_data, learning_rate, momentum, epoch, use_zero, max_order, 5, mmd, MMD_RBF_main, MMDweight, re_save_path, device)

	plt.plot(e1, p1, 'r--', e2, p2, 'b--')
	plt.xlabel('Step', fontsize=12)
	plt.ylabel('$PD_{dist}$', fontsize=12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.legend(["MMD", "MMD + PPM-Reg"], loc="upper right", fontsize=12)	
	plt.xlim([0, 20*120])
	plt.savefig(save_path + 'PD.png', bbox_inches='tight')
	plt.close()








