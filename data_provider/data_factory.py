import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import umap
import umap.plot

from utils.functions import Ornstein_Uhlenbeck, region_to_number


# these classes are for liberty in the preprocessing of the raw data, what do we have access to, how the graph is infered from predictors


class bdclim:
    def __init__(self, root_path, data_path='bdclim_safran_2022-2024.nc'):
        """
        Initialize a time series dataset from a xarray dataset.

        :param dataset: dataframe containing the data, shape: n_steps, n_nodes
        :param mask: mask for valid data (1:valid, 0:not valid)
        """
        super().__init__()

        # load dataset
        self.dataset = xr.load_dataset(os.path.join(root_path, data_path))

        # set dataset dataframe
        self.df = self.dataset.reset_coords()['t'].to_pandas()

        # set exogenous variables (predictors) dataframe
        self.predictors = self.dataset.reset_coords().drop_vars(['t','Station_Name','type_temps','reseau_poste_actuel','lat','lon']).isel(time=0).to_dataframe().drop(columns='time')

        mask = (~np.isnan(self.df.values)).astype('uint8')
        self.mask = mask

    def correlation_adjacency(self, threshold=0.1):
        corr_matrix = self.df.corr()
        corr_matrix[corr_matrix < threshold] = 0
        corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))
        return corr_matrix.values
    
    def umap_adjacency(self, threshold=0.1, verbose=False):
        reducer = umap.UMAP(min_dist=0.9, n_neighbors=10, metric='euclidean')
        embedding = reducer.fit_transform(self.predictors.drop(columns='region').fillna(method='ffill'))

        adjacency_matrix = reducer.graph_.toarray()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))

        if verbose:
            umap.plot.points(reducer, labels=region_to_number(self.predictors['region']))
            umap.plot.connectivity(reducer, show_points=True, edge_bundling='hammer', labels=region_to_number(self.predictors['region']))
        return adjacency_matrix

    def __repr__(self):
        return "{}(nodes={}, length={})".format(self.__class__.__name__, self.n_nodes, self.__len__())
    
    def __len__(self):
        return self.df.values.shape[0]
    
    @property
    def n_nodes(self):
        return self.df.values.shape[1]

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def numpy(self, return_idx=False):
        if return_idx:
            return self.df.values, self.df.index
        return self.df.values

    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)
            

class bdclim_clean:
    def __init__(self, root_path, data_path='bdclim_safran_2022-2024.nc'):
        """
        Initialize a time series dataset from a xarray dataset.

        :param dataset: dataframe containing the data, shape: n_steps, n_nodes
        :param mask: mask for valid data (1:valid, 0:not valid)
        """
        super().__init__()

        # load dataset
        self.dataset = xr.load_dataset(os.path.join(root_path, data_path))

        # drop NaN values
        self.dataset = self.dataset.dropna(dim='num_poste')

        # set dataset dataframe
        self.df = self.dataset.reset_coords()['t'].to_pandas()

        # set optional exogenous variables (predictors) dataframe
        self.predictors = self.dataset.reset_coords().drop_vars(['t','Station_Name','type_temps','reseau_poste_actuel','lat','lon']).isel(time=0).to_dataframe().drop(columns='time')

        mask = (~np.isnan(self.df.values)).astype('uint8')
        self.mask = mask
    
    def correlation_adjacency(self, threshold=0.1):
        corr_matrix = self.df.corr()
        corr_matrix[corr_matrix < threshold] = 0
        corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))
        return corr_matrix.values
    
    def umap_adjacency(self, threshold=0.1, verbose=False):
        reducer = umap.UMAP(min_dist=0.9, n_neighbors=10, metric='euclidean')
        reducer.fit_transform(self.predictors.drop(columns='region').fillna(method='ffill'))

        adjacency_matrix = reducer.graph_.toarray()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))

        if verbose:
            umap.plot.points(reducer, labels=region_to_number(self.predictors['region']))
            umap.plot.connectivity(reducer, show_points=True, edge_bundling='hammer', labels=region_to_number(self.predictors['region']))
        return adjacency_matrix
        
    def __repr__(self):
        return "{}(nodes={}, length={})".format(self.__class__.__name__, self.n_nodes, self.__len__())
    
    def __len__(self):
        return self.df.values.shape[0]
    
    @property
    def n_nodes(self):
        return self.df.values.shape[1]
    
    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()
    
    def numpy(self, return_idx=False):
        if return_idx:
            return self.df.values, self.df.index
        return self.df.values

    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)

    
# class bdclim_clean_sequence_mask(Dataset):
#     def __init__(self, args, root_path, flag='train', data_path='bdclim_safran_2022-2024.nc', scale=False):
#         self.args = args
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.flag = type_map[flag]
#         self.root_path = root_path
#         self.data_path = data_path
#         self.scale = scale
#         self.__read_data__()

#     def __read_data__(self):
#         data = xr.load_dataarray(os.path.join(self.root_path,self.data_path))
#         train_data_array = data.dropna(dim='num_poste')
#         X= train_data_array.values
#         self.seq_len = X.shape[0]
#         self.seq_dim = X.shape[1]
#         # print('seq_len:', self.seq_len)
#         # print('seq_dim:', self.seq_dim)

#         num_train = int(self.seq_len * 0.7)
#         num_test = int(self.seq_len * 0.2)
#         num_vali = self.seq_len - num_train - num_test
#         border1s = [0, num_train, num_train + num_test]
#         border2s = [num_train, num_train + num_test, num_train + num_test + num_vali]
#         border1 = border1s[self.flag]
#         border2 = border2s[self.flag]
        
#         # mask = np.ones((self.seq_len, self.seq_dim), dtype=bool)
#         # mask[np.random.rand(self.seq_len, self.seq_dim) < self.args.mask_proba] = 0
#         mask = np.array(np.random.rand(self.seq_len, self.seq_dim) < self.args.mask_proba/self.args.mask_len, dtype=bool)
#         indices =  np.where(mask)
#         for i,j in zip(indices[0], indices[1]):
#             start = max(0, i - int(0.5 * self.args.mask_len))
#             end = min(self.seq_len, i + int(0.5 * self.args.mask_len))
#             mask[start:end, j] = True
#         data_X = X.copy()
#         data_X[mask] = np.NaN
#         data_y = X.copy()

#         # self.data_X = data_X[border1:border2]
#         # self.data_y = data_y[border1:border2]
#         self.data_X = torch.tensor(data_X[border1:border2]).float()
#         self.data_y = torch.tensor(data_y[border1:border2]).float()
#         #self.mask = mask[border1:border2] should we pass the mask? why don't they in sequence setting ?

#     def __getitem__(self, index):
#         sample = (self.data_X[index], self.data_y[index])

#         if self.scale:
#             self.scaler = StandardScaler()
#             self.scaler.fit(sample[0])
#             sample = (self.scaler.transform(sample[0]), sample[1])
#         return sample
    
#     def __len__(self):
#         return [int(self.seq_len * 0.7), int(self.seq_len * 0.2), int(self.seq_len * 0.1)][self.flag]

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    

