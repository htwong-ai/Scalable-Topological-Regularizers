import numpy as np
import torch
import ot
import embedding
import distance
from torchvision import datasets, transforms

embedding_model = embedding.ClipEmbeddingModel()


def ComputeQuality(GenModel, batch_size, target_I, device, RealDataSet, latent_dim, VRdim = 0):

	train_loader = torch.utils.data.DataLoader(RealDataSet, batch_size = batch_size, shuffle = False, drop_last = True, pin_memory = True, num_workers = 1)
	start_idx = 0
	embedding_model = embedding.ClipEmbeddingModel()
	dims = 768

	ValueStoreR = np.zeros((batch_size*target_I, dims))
	ValueStoreF = np.zeros((batch_size*target_I, dims))

	for i,  (data, label) in enumerate(train_loader):
		if(i + 1 > target_I):
			break

		with torch.no_grad():
			GenData = GenModel((torch.rand(batch_size, latent_dim)*2-1).to(device))
			GenData = GenData.transpose(1,3)
			data = data.transpose(1,3)
			ValueStoreR[start_idx:start_idx + batch_size] = embedding_model.embed(data.numpy())
			ValueStoreF[start_idx:start_idx + batch_size] = embedding_model.embed(GenData.cpu().numpy()) 

			start_idx = start_idx + batch_size

	with torch.no_grad():
		Cmmd = distance.mmd(ValueStoreR, ValueStoreF)

		ValueStoreR = torch.from_numpy(ValueStoreR)
		ValueStoreF = torch.from_numpy(ValueStoreF)

		RdistMatrix = torch.cdist(ValueStoreR, ValueStoreR)
		FdistMatrix = torch.cdist(ValueStoreF, ValueStoreF)

		
	with torch.no_grad():

		M = torch.cdist(ValueStoreR.to(device), ValueStoreF.to(device))
		a = (torch.ones(ValueStoreR.size()[0])/ValueStoreR.size()[0]).to(device)
		b = (torch.ones(ValueStoreR.size()[0])/ValueStoreR.size()[0]).to(device)

		T = ot.emd(a, b, M)
		WD_dist = torch.sqrt((T*M).sum())

	return Cmmd, WD_dist


