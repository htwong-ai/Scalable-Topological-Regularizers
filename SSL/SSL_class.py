import argparse

def parse_args():
    desc = "UnsuperLearn"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'kmnist'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=2048, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=96, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--model_path', default = "CelebA", type=str, help='Select_model_dir')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--num_labels', type=int, default=100)
    parser.add_argument('--output_dim', type=int, default=64)

    return parser.parse_args()



import torch
import os, cv2, math, random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from csv import writer
from Model import gModel, dModel, f2cModel
from dataloader import dataloader

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

opt = parse_args()

path = opt.model_path
batch_size = opt.batch_size

transform = transforms.Compose([transforms.ToTensor()])

if opt.dataset == 'mnist':
    training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

elif opt.dataset == 'fmnist':
    training_set = datasets.FashionMNIST('data/Fmnist', train=True, download=True, transform=transform)
    #test_set = datasets.FashionMNIST('data/Fmnist', train=False, download=True, transform=transform)

elif opt.dataset == 'kmnist':
    training_set = datasets.KMNIST('data/Kmnist', train=True, download=True, transform=transform)

d = dModel(out_dim = 64)

checkpoint_d = torch.load(opt.model_path + "/4000d_dev.pth")
d.load_state_dict(checkpoint_d['model_state_dict'])

FileDir = opt.model_path + '/acc_'+ str(opt.num_labels) +'.csv'

d.to(device)
result = np.zeros(10)

for copy_run in range(10):

    c = f2cModel(512)
    c.to(device)
    labeled_loader, unlabeled_loader, test_loader = dataloader(opt.dataset, opt.input_size, batch_size*2, opt.num_labels)
    C_optimizer = torch.optim.Adam(c.parameters(), lr=opt.lr, betas=(0.1, 0.99))

    print("Run:" +  str(copy_run))
    max_acc = 0

    for epochs in range(opt.epoch):

        for iter, (x_u, y_u) in enumerate(labeled_loader):

            f = d(x_u.to(device))
            
            c2 = c(f)
                
            C_labeled_loss = torch.nn.functional.nll_loss(c2, y_u.to(device))

            C_optimizer.zero_grad(set_to_none=True)
            C_labeled_loss.backward()
            C_optimizer.step()

        if (epochs % 20 == 0):
            correct = 0
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    f = d(data)
                    c2 = c(f)
                    pred = c2.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            acc = 100. * correct / len(test_loader.dataset)
            
            if(acc > max_acc):
                max_acc = acc
                #print(max_acc)

    result[copy_run] = max_acc
    DataList = [ copy_run, max_acc ]

    with open(FileDir, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(DataList)
        f_object.close()

print("The average accuracy is: " + str(np.mean(result)))
print("The standard deviation is: " + str(np.std(result)))

