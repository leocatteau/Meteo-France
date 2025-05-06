import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree


class mean_fill(nn.Module):
    def __init__(self, columnwise=False):
        super(mean_fill, self).__init__()
        self.columnwise = columnwise

    def forward(self, x, mask):
        if self.columnwise: 
            x_pred = torch.nanmean(x,dim=2, keepdim=True)*(~mask) + torch.nan_to_num(x)
        else: x_pred = torch.nanmean(x)*(~mask) + torch.nan_to_num(x)
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
            plt.title('Rank optimization for SVD')
            plt.grid()
            plt.show()

    def predict(self, x):
        mask = torch.isnan(x)

        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)

        U, S, V = np.linalg.svd(x_mean, full_matrices=False)
        U, S, V = U[:,:self.rank], np.diag(S[:self.rank]), V[:self.rank,:]
        x_pred = torch.tensor(U @ S @ V).float()
        x_pred = x_pred*(mask) + torch.nan_to_num(x)

        return x_pred
    
    
class random_forest():
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    
    def train(self, x, y, verbose=False):
        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)
        self.model.fit(x_mean, y)

        if verbose:
            tree = self.model.estimators_[0]
            plt.figure(figsize=(20, 10))
            plot_tree(tree, filled=True, rounded=True)
            plt.title("Decision Tree from the Random Forest")
            plt.show()
            

    def predict(self, x):
        mask = torch.isnan(x)

        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)

        y_pred = torch.tensor(self.model.predict(x_mean)).float()
        y_pred = y_pred*(mask) + np.nan_to_num(x)
        return y_pred
    
    
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