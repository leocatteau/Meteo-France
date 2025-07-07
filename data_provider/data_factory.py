import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
# import umap
# import umap.plot
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph

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
        print("before loading dataset, path: ", os.path.join(root_path, data_path))
        self.dataset = xr.load_dataset(os.path.join(root_path, data_path))
        print("dataset loaded")

        # set dataset dataframe
        self.df = self.dataset.reset_coords()['t'].to_pandas()

        # drop stations with only NaN values
        total_stations = self.df.shape[1]
        self.df = self.df.dropna(axis=1, how='all')
        print("total stations: ", total_stations, " remaining stations: ", self.df.shape[1], " removing stations with only NaN values.")

        # # drop stations with less than 5 years data
        # threshold = 5 * 365 * 24
        # self.df = self.df.loc[:, self.df.count() >= threshold]
        # print("remaining stations after 5 years threshold: ", self.df.shape[1])

        # set optional exogenous variables (predictors) dataframe
        self.exogenous_vars = self.dataset.reset_coords()['type_temps'].to_pandas().reset_index(drop=True).values
        self.exogenous_vars = np.transpose(np.tile(self.exogenous_vars, (self.df.shape[1], 1)))
        self.predictors = self.dataset.reset_coords().drop_vars(['t','Station_Name','reseau_poste_actuel','lat','lon', 'type_temps']).isel(time=0).to_dataframe().drop(columns='time')

        mask = (~np.isnan(self.df.values)).astype('uint8')
        self.mask = mask

    def correlation_adjacency(self, threshold=0.1, verbose=False):
        corr_matrix = self.df.corr()
        corr_matrix[corr_matrix < threshold] = 0
        corr_matrix.fillna(0, inplace=True)
        corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))
        if verbose:
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.title('Correlation Matrix')
            plt.show()

            G = nx.from_numpy_array(corr_matrix.values)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            nx.draw_networkx(G, with_labels=False, node_size=3, width=0.5, ax=ax)
            plt.title('Correlation Network')
            plt.show()

        return corr_matrix.values
    
    # def umap_adjacency(self, threshold=0.1, verbose=False):
    #     predictors = (self.predictors.drop(columns='region') - self.predictors.drop(columns='region').mean()) / self.predictors.drop(columns='region').std()
    #     predictors = self.predictors.drop(columns='region')
    #     reducer = umap.UMAP(min_dist=0.5, n_neighbors=10, metric='euclidean')
    #     reducer.fit_transform(predictors.fillna(method='ffill'))

    #     adjacency_matrix = reducer.graph_.toarray()
    #     adjacency_matrix[adjacency_matrix < threshold] = 0
    #     adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))

    #     if verbose:
    #         umap.plot.points(reducer, labels=self.predictors['region'])
    #         umap.plot.connectivity(reducer, show_points=True, edge_bundling='hammer')
    #         plt.title('Infered graph from predictors')
    #         plt.show()
    #     return adjacency_matrix
    
    def KNN_adjacency(self, threshold=0.1, verbose=False):
        predictors = (self.predictors.drop(columns='region') - self.predictors.drop(columns='region').mean()) / self.predictors.drop(columns='region').std()
        predictors = self.predictors.drop(columns='region')

        adjacency_matrix = kneighbors_graph(predictors.fillna(method='ffill'), n_neighbors=10, mode='connectivity', include_self=False).toarray()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
        G = nx.from_numpy_array(adjacency_matrix)

        if verbose:
            plt.figure(figsize=(10, 8))
            plt.imshow(adjacency_matrix, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.title('KNN Adjacency Matrix')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            nx.draw_networkx(G, with_labels=False, node_size=3, width=0.5, ax=ax)
            plt.title('KNN Adjacency Network')
            plt.show()
        return adjacency_matrix, G

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

    def pytorch(self, return_idx=False):
        if return_idx:
            return torch.FloatTensor(self.df.values), self.df.index
        return torch.FloatTensor(self.df.values)
            

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
        total_stations = self.dataset.sizes['num_poste']
        self.dataset = self.dataset.dropna(dim='num_poste')
        print("total stations: ", total_stations, " remaining stations: ", self.dataset.sizes['num_poste'], " removing every station with NaN values.")

        # set dataset dataframe
        self.df = self.dataset.reset_coords()['t'].to_pandas()

        # set optional exogenous variables (predictors) dataframe
        self.exogenous_vars = self.dataset.reset_coords()['type_temps'].to_pandas().reset_index(drop=True).values
        self.exogenous_vars = np.transpose(np.tile(self.exogenous_vars, (self.df.shape[1], 1)))
        self.predictors = self.dataset.reset_coords().drop_vars(['t','Station_Name','reseau_poste_actuel','lat','lon', 'type_temps']).isel(time=0).to_dataframe().drop(columns='time')

        mask = (~np.isnan(self.df.values)).astype('uint8')
        self.mask = mask
    
    def correlation_adjacency(self, threshold=0.1, verbose=False):
        corr_matrix = self.df.corr()
        corr_matrix[corr_matrix < threshold] = 0
        corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))
        if verbose:
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.title('Correlation Matrix')
            plt.show()

            G = nx.from_numpy_array(corr_matrix.values)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            nx.draw_networkx(G, with_labels=False, node_size=3, width=0.5, ax=ax, labels=region_to_number(self.predictors['region']))
            plt.legend()
            plt.title('Correlation Network')
            plt.show()

        return corr_matrix.values
    
    # def umap_adjacency(self, threshold=0.1, verbose=False):
    #     predictors = (self.predictors.drop(columns='region') - self.predictors.drop(columns='region').mean()) / self.predictors.drop(columns='region').std()
    #     predictors = self.predictors.drop(columns='region')
    #     reducer = umap.UMAP(min_dist=1, n_neighbors=50, metric='euclidean')
    #     reducer.fit_transform(predictors.fillna(method='ffill'))

    #     adjacency_matrix = reducer.graph_.toarray()
    #     adjacency_matrix[adjacency_matrix < threshold] = 0
    #     adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))

    #     if verbose:
    #         umap.plot.points(reducer, labels=self.predictors['region'])
    #         umap.plot.connectivity(reducer, show_points=True, edge_bundling='hammer', labels=region_to_number(self.predictors['region']))
    #     return adjacency_matrix
    
    def KNN_adjacency(self, threshold=0.1, verbose=False):
        predictors = (self.predictors.drop(columns='region') - self.predictors.drop(columns='region').mean()) / self.predictors.drop(columns='region').std()
        predictors = self.predictors.drop(columns='region')

        adjacency_matrix = kneighbors_graph(predictors.fillna(method='ffill'), n_neighbors=10, mode='connectivity', include_self=False).toarray()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
        G = nx.from_numpy_array(adjacency_matrix)

        if verbose:
            plt.figure(figsize=(10, 8))
            plt.imshow(adjacency_matrix, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.title('KNN Adjacency Matrix')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            nx.draw_networkx(G, with_labels=False, node_size=3, width=0.5, ax=ax)
            plt.title('KNN Adjacency Network')
            plt.show()
        return adjacency_matrix, G
        
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

    def pytorch(self, return_idx=False):
        if return_idx:
            return torch.FloatTensor(self.df.values), self.df.index
        return torch.FloatTensor(self.df.values)