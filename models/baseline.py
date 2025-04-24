import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from utils.functions import torch_nanmean, torch_nan_to_num


class mean_fill(nn.Module):
    def __init__(self, columnwise=False):
        super(mean_fill, self).__init__()
        self.columnwise = columnwise

    def forward(self, x):
        mask = torch.isnan(x)
        if self.columnwise: 
            x_pred = torch_nanmean(x,dim=1)[:,None]*mask + torch_nan_to_num(x)
        else: x_pred = torch_nanmean(x)*mask + torch_nan_to_num(x)
        return x_pred
    
    
class svd():
    def __init__(self, rank=1):
        self.rank = rank

    def train(self, x, y, verbose=False):
        seq_len, seq_dim = x.shape[0], x.shape[1]
        MSE_svd = []
        ranks = np.arange(1, min(seq_len, seq_dim), 10)
        for rank in ranks:
            self.rank = rank
            x_pred = self.predict(x)
            MSE_svd.append(mean_squared_error(y, x_pred))
        MSE_svd = np.array(MSE_svd)
        optimal_rank = ranks[np.argmin(MSE_svd)]
        self.rank = optimal_rank

        if verbose:
            print('optimal rank:', optimal_rank)
            mean_model = mean_fill(columnwise=True)
            x_mean = mean_model(x)
            MSE_mean = mean_squared_error(y, x_mean)

            plt.figure(figsize=(8, 4))
            plt.plot(ranks, MSE_svd, label='svd reconstruction')
            plt.axvline(optimal_rank, color='r', linestyle='--', label='optimal rank')
            plt.axhline(MSE_mean, linestyle='--', label='mean reconstruction')
            plt.xlabel('rank')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    def predict(self, x):
        mask = torch.isnan(x)

        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)

        U, S, V = np.linalg.svd(x_mean, full_matrices=False)
        U, S, V = U[:,:self.rank], np.diag(S[:self.rank]), V[:self.rank,:]
        x_pred = torch.tensor(U @ S @ V).float()
        x_pred = x_pred*(mask) + torch_nan_to_num(x)

        return x_pred
    
    
class linear():
    def __init__(self):
        self.model = LinearRegression()

    def train(self, x, y):
        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)
        self.model.fit(x_mean, y)

    def predict(self, x):
        mask = torch.isnan(x)

        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)

        y_pred = torch.tensor(self.model.predict(x_mean)).float()
        y_pred = y_pred*(mask) + np.nan_to_num(x)
        return y_pred
    

class random_forest():
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10)
    
    def train(self, x, y):
        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)
        self.model.fit(x_mean, y)

    def predict(self, x):
        mask = torch.isnan(x)

        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)

        y_pred = torch.tensor(self.model.predict(x_mean)).float()
        y_pred = y_pred*(mask) + np.nan_to_num(x)
        return y_pred