import torch
import numpy as np
import itertools
import random

def generate_subsamples(N, num_ppm, num_samples, use_zero, seed = None, device = "cpu"):
    subsamples = []
    min_order = 1
    if use_zero:
        min_order = 0
    for i in range(num_ppm):
        cur_order = min_order + i
        if(seed == None):
            subsamples.append(torch.randint(0,N, (num_samples[i], 2*cur_order+2)))
        else:
            #print(torch.randint(0,N, (num_samples[i], 2*cur_order+2), generator = torch.Generator().manual_seed(seed)))
            subsamples.append(torch.randint(0,N, (num_samples[i], 2*cur_order+2), generator = torch.Generator().manual_seed(seed)))

    return subsamples

def compute_simple_homology(M):
    num_samples, num_pts, _ = M.size()

    # In the case where there are only 2 points, homology is just given by pairwise distances
    if num_pts == 2:
        return M[:,:,0]

    # Remove the zero entry from each row of the distance matrix
    M_reduced = M.flatten(start_dim = 1)[:,1:].view(num_samples, num_pts-1, num_pts+1 )[:,:,:-1].reshape(num_samples, num_pts, num_pts-1)

    # Find top two distances in each row (corresponding to x^(1), x^(2))
    M_sort, _ = torch.topk(M_reduced, 2)

    # Compute t_b and t_d
    tb, _ = torch.max(M_sort[:,:,1], dim = 1, keepdim=True)
    td, _ = torch.min(M_sort[:,:,0], dim = 1, keepdim=True)

    # Concatenate into array
    check = tb < td
    tb_out = tb[check]
    td_out = td[check]
    out = torch.cat((tb_out.unsqueeze(1), td_out.unsqueeze(1)), dim=1)

    return out

def SamplePa(N, MaxOrder, Sample, PaGen, UseZero=False, device = "cpu"):
    P = []
    k = 0
    if(UseZero):
        it = [random.sample(range(N), 2) for j in range(int(Sample[k])*PaGen)]
        P.append(torch.from_numpy(np.array(it)).to(device))
        k = k + 1
    for i in range(1, MaxOrder+1):
        it = [random.sample(range(N), 2*i+2) for j in range(int(Sample[k])*PaGen)]
        #print(it)
        P.append(torch.from_numpy(np.array(it)).to(device))
        k = k + 1
    return P

# num_samples is either a list or a number
def compute_ppm(X, max_order, num_samples, precompute = False, pa = None, use_zero=False, seed = None, device = "cpu"):
    N, _ = X.size()
    X_distance_matrix = torch.cdist(X, X, p=2)

    # Initialize list of diagrams
    D = []

    # Total number of PPMs
    num_ppm = max_order
    if use_zero:
        num_ppm = max_order + 1

    # If num samples is a number, turn it into a list
    if not type(num_samples) == list:
        num_samples = [num_samples for i in range(num_ppm)]

    # Generate the required subsamples if we have not precomputed them
    if not precompute:
        subsamples = generate_subsamples(N, num_ppm, num_samples, use_zero, seed, device)
    
    # Compute homology
    for i in range(num_ppm):
        
        if(precompute):
            tep = pa[i]
            index = torch.randint(0, pa[i].size()[0], ( int(num_samples[i]) ,1)).squeeze()
            cur_subsample = tep[index,:]
        else:
            cur_subsample = subsamples[i]

        M = X_distance_matrix[cur_subsample[:,:,None], cur_subsample[:,None]]
        D.append(compute_simple_homology(M))

    return D


class MMD(torch.nn.Module):

    def __init__(self, width, weights = None, num_samples = None, decay_exponent = 1.0, device = 'cuda'):
        super().__init__()
        self.width = width
        self.weights = weights
        self.num_samples = num_samples
        self.decay_exponent = decay_exponent

    # Computes unnormalized MMD using RBF kernel including lifetime decay
    # NOTE: This assumes that number of samples used for X and Y are the same (though
    #       number of points in each diagram may differ)
    def MMD_RBF_single(self, X, Y):

        # If Y is empty, then just return the unnormalized MMD of X to empty measure
        if Y == None:
            XX_dist = torch.cdist(X,X,p=2)
            K_XX = torch.exp(-torch.pow(XX_dist,2)/self.width)
            X_lt_decay = torch.pow(X[:,1],self.decay_exponent)
            K_XX_decay = K_XX * X_lt_decay[:, None] * X_lt_decay[None,:]
            return K_XX_decay.sum()

        # Compute usual RBF kernel matrix
        XX_dist = torch.cdist(X,X,p=2)
        YY_dist = torch.cdist(Y,Y,p=2)
        XY_dist = torch.cdist(X,Y,p=2)

        K_XX = torch.exp(-torch.pow(XX_dist,2)/self.width)
        K_YY = torch.exp(-torch.pow(YY_dist,2)/self.width)
        K_XY = torch.exp(-torch.pow(XY_dist,2)/self.width)

        # Incorporate lifetime decays
        X_lt_decay = torch.pow(X[:,1],self.decay_exponent)
        Y_lt_decay = torch.pow(Y[:,1],self.decay_exponent)

        K_XX_decay = K_XX * X_lt_decay[:, None] * X_lt_decay[None,:]
        K_YY_decay = K_YY * Y_lt_decay[:, None] * Y_lt_decay[None,:]
        K_XY_decay = K_XY * X_lt_decay[:, None] * Y_lt_decay[None,:]

        # Return unnormalized MMD
        return K_XX_decay.sum() - 2*K_XY_decay.sum() + K_YY_decay.sum()
    
    def forward(self, X, Y):

        MMD_all = torch.zeros(len(X), device=X[0].device)

        for i in range(len(X)):
            DX = X[i]
            DY = Y[i]

            num_X = len(DX)
            num_Y = len(DY)

            # If both diagrams are empty, then distance is trivial
            # If one of the diagrams is empty, then MMD is just equal to one of th
            # Note: Here we transform (birth, death) coordinates into (birth, lifetime)
            if ( (num_X==0) | (num_Y==0) ):
                if (num_X == num_Y):
                    MMD_all[i] = 0.0
                elif num_X == 0:
                    DY_lt = torch.cat((DY[:,0].unsqueeze(1), (DY[:,1] - DY[:,0]).unsqueeze(1)), 1)
                    MMD_all[i] = self.MMD_RBF_single(DY_lt, None)*float(self.weights[i]/(self.num_samples[i]**2))
                elif num_Y == 0:
                    DX_lt = torch.cat((DX[:,0].unsqueeze(1), (DX[:,1] - DX[:,0]).unsqueeze(1)), 1)
                    MMD_all[i] = self.MMD_RBF_single(DX_lt, None)*float(self.weights[i]/(self.num_samples[i]**2))
            else:
                DX_lt = torch.cat((DX[:,0].unsqueeze(1), (DX[:,1] - DX[:,0]).unsqueeze(1)), 1)
                DY_lt = torch.cat((DY[:,0].unsqueeze(1), (DY[:,1] - DY[:,0]).unsqueeze(1)), 1)
                MMD_all[i] = self.MMD_RBF_single(DX_lt, DY_lt)*float(self.weights[i]/(self.num_samples[i]**2))

        return MMD_all.sum()


def MMD_RBF_main(X, Y, width=0.1):

    nX = torch.tensor(X.size()[0]).float()
    nY = torch.tensor(Y.size()[0]).float()

    # Compute usual RBF kernel matrix
    XX_dist = torch.cdist(X,X,p=2)
    YY_dist = torch.cdist(Y,Y,p=2)
    XY_dist = torch.cdist(X,Y,p=2)

    K_XX = torch.exp(-torch.pow(XX_dist,2)/width)
    K_YY = torch.exp(-torch.pow(YY_dist,2)/width)
    K_XY = torch.exp(-torch.pow(XY_dist,2)/width)

    # Return unnormalized MMD
    return (1/torch.pow(nX,2))*K_XX.sum() - (2/nX/nY)*K_XY.sum() + (1/torch.pow(nY,2))*K_YY.sum()