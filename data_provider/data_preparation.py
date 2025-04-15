import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset


# these classes are the training preparation layers over the raw datasets, for liberty in masking processes, etc

class WindowHorizonDataset(Dataset):
    def __init__(self,args,data_source):
        super(WindowHorizonDataset, self).__init__()
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.window = args.window
        self.horizon = args.horizon
        self.scaler = args.scaler
        self.mask = data_source.mask
        self.coarse_frequency = self.window + self.horizon

        # artificial evaluation masking
        eval_mask = np.array(np.random.rand(len(data_source), data_source.n_nodes) > args.mask_proba/self.window, dtype=bool)
        masked_indices =  np.where(~eval_mask)
        for i,j in zip(masked_indices[0], masked_indices[1]):
            start = max(0, i - int(0.5 * self.window))
            end = min(len(data_source), i + int(0.5 * self.window))
            eval_mask[start:end, j] = False
        self.mask = eval_mask
        self.eval_mask = eval_mask
        

    def __getitem__(self, index):
        #sample = (self.data[index:index+self.window],self.data[index+self.window:index+self.window+self.horizon])
        # shouldn't we introduce a mask? and give data from other stations in prediction time window ? 
        sample = dict()
        sample['x'] = self.data[index:index+self.window][:,:,None]
        # sample['x'][~self.mask[index:index+self.window]] = np.nan
        # sample['y'] = self.data[index+self.window:index+self.window+self.horizon][:,:,None]
        sample['y'] = self.data[index:index+self.window][:,:,None]
        sample['mask'] = self.mask[index:index+self.window][:,:,None]
        sample['eval_mask'] = self.eval_mask[index:index+self.window][:,:,None]
        if self.scaler is not None:
            raise NotImplementedError("scaler not implemented yet.")
        return sample
    
    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self))
    
    def dataframe(self):
        return pd.DataFrame(data=self.data, index=self.indices)
    
    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)

    
class SequenceMaskDataset(Dataset):
    def __init__(self,args,data_source):
        super(SequenceMaskDataset, self).__init__()
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.window = args.window
        self.scaler = args.scaler
        self.mask = data_source.mask
        self.coarse_frequency = 1

        # artificial masking
        train_mask = np.array(np.random.rand(len(data_source), data_source.n_nodes) > args.mask_proba/self.window, dtype=bool)
        masked_indices =  np.where(~train_mask)
        for i,j in zip(masked_indices[0], masked_indices[1]):
            start = max(0, i - int(0.5 * self.window))
            end = min(len(data_source), i + int(0.5 * self.window))
            train_mask[start:end, j] = False
        # train_mask[np.where(self.mask)] = True # eviter l'entrainement sur les valeures masquées à débugger
        self.train_mask = train_mask
        self.corrupted_data = self.data.copy()
        self.corrupted_data[~self.train_mask] = np.nan

    def __getitem__(self, index):
        sample = (self.corrupted_data[index], self.data[index])
        
        if self.scaler is not None:
            raise NotImplementedError("scaler not implemented yet.")
        return sample
    
    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self))
    
    def dataframe(self):
        return pd.DataFrame(data=self.data, index=self.indices)
    
    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)
    

class SampleMaskDataset(Dataset):
    def __init__(self,args,data_source):
        super(SampleMaskDataset, self).__init__()
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.scaler = args.scaler
        self.mask = data_source.mask
        self.coarse_frequency = 1

        # artificial masking
        self.train_mask = np.zeros((len(data_source), data_source.n_nodes), dtype=bool)
        self.train_mask[np.random.rand(len(data_source), data_source.n_nodes) > args.mask_proba] = 1
        self.corrupted_data = self.data.copy()
        self.corrupted_data[~self.train_mask] = np.nan

    def __getitem__(self, index):
        sample = (self.corrupted_data[index], self.data[index])
        
        if self.scaler is not None:
            raise NotImplementedError("scaler not implemented yet.")
        return sample
    
    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self))
    
    def dataframe(self):
        return pd.DataFrame(data=self.data, index=self.indices)
    
    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)