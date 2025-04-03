import numpy as np
import torch


def Ornstein_Uhlenbeck(d, N, dt, X0, mu, theta, sigma=False):
    if sigma==False:
        seed = np.random.RandomState(4)
        S = seed.randn(d,d*100)/np.sqrt(d*100)  
        sigma = S@S.T
    X = np.zeros((N, d))
    X[0] = X0
    for t in range(1, N): 
        dW = np.sqrt(dt) * np.random.multivariate_normal(np.zeros(d), sigma)
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma @ dW
    return np.array(X), sigma

def torch_nanmean(tensor, dim=None):
    '''this function exists in pytorch >= 1.8.0'''
    mask = ~torch.isnan(tensor)
    tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device))  
    if dim is not None:
        sum_ = torch.sum(tensor, dim=dim)
        count = torch.sum(mask, dim=dim)
    else:
        sum_ = torch.sum(tensor)
        count = torch.sum(mask)
    return sum_ / count 

def torch_nan_to_num(tensor, nan=0.0):
    return torch.where(torch.isnan(tensor), torch.tensor(nan, device=tensor.device), tensor)

def region_to_number(regions):
    dict_region = {}
    for i, region in enumerate(np.unique(regions)):
        dict_region[region] = i
    return np.array([dict_region[region] for region in regions])

