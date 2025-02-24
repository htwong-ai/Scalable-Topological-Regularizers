import torch
import argparse
import os
import numpy as np

from torchvision import datasets, transforms
from model import Generator32, ML32
from ComputeQualityWD import ComputeQuality
from csv import writer



parser = argparse.ArgumentParser(description='Compute Metric')
parser.add_argument('--data', default = "AnimeFace", type=str, help='Select Data Set')
parser.add_argument('--data_path', default = "/home/user/Downloads/Img", type=str, help='Select Data Set path')
parser.add_argument('--model_path', default = "AnimeFace", type=str, help='Select_model_dir')
parser.add_argument('--out_path', default = "Pre_AnimeFace_TDR_1.0_Order_1_Width1.0_usezero_Sample_1024_Sample_1024_weight_0.001_weight_0.6", type=str, help='Select_output_dir')


opt = parser.parse_args()

FileDir = opt.out_path + '/metric.csv'

out_dim = 128

netG = Generator32()  
netML = ML32(out_dim=out_dim)

data_path = opt.data_path
Size = 32

transform = transforms.Compose([
        transforms.Resize(Size),
        transforms.CenterCrop(Size),
        transforms.ToTensor()
    ])

#FileDir = 'metric2.csv'

if(opt.data == "CelebA"):
    CelebA = datasets.CelebA(root = data_path, split = "train", transform = transform, download=True)    
elif(opt.data == "AnimeFace"):
	AnimeFace = datasets.ImageFolder(root = data_path, transform = transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('loading model')

G_load = torch.load(opt.model_path)
netG.load_state_dict(G_load["G_State_Dict"])
netG = netG.to(device)

with torch.no_grad():
	if(opt.data == "CelebA"):
		CMMD, WD = ComputeQuality(netG, 100, 100, "cuda:0", CelebA, out_dim)
	elif(opt.data == "AnimeFace"):
		CMMD, WD = ComputeQuality(netG, 100, 100, "cuda:0", AnimeFace, out_dim)
            

	print("The CMMD is: " + str(CMMD.cpu().numpy()))
	print("The WD is: " + str(WD.cpu().numpy()))
		
	DataList = [ CMMD.cpu().numpy(), WD.cpu().numpy()]
	with open(FileDir, 'a') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow(DataList)
		f_object.close()