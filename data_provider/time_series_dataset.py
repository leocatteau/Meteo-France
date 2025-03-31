import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, indices=None,scaler=None,window=24,horizon=24):
        super(TimeSeriesDataset, self).__init__()
        self.data = data
        self.indices = indices
        self.freq = indices.inferred_freq
        self.freq = pd.tseries.frequencies.to_offset(self.freq)
        self.window = window
        self.horizon = horizon
        self.scaler = scaler

    def __getitem__(self, index):
        index = self.indices[index]
        sample = dict()
        # shouldn't we introduce a mask? and give data from other stations in prediction time window ? 
        sample['x'] = self.data[index:index+self.window]
        sample['y'] = self.data[index+self.window:index+self.window+self.horizon]
        if self.scaler is not None:
            raise NotImplementedError("scaler not implemented yet.")
        return sample
    
    def __len__(self):
        return len(self._indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self))
    
    def dataframe(self):
        return pd.DataFrame(data=self.data, index=self.index)