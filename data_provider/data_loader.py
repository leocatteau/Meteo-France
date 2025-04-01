import numpy as np
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import bdclim_window_horizon
from data_provider.time_series_dataset import TimeSeriesDataset

data_dict = {
    'bdclim_window_horizon': bdclim_window_horizon,
    # 'Ornstein_Uhlenbeck': Ornstein_Uhlenbeck_Data,
    # more to be added
}

class DataProvider:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        Data = data_dict[args.data]
        self.data = Data(root_path=args.root_path,
                         data_path=args.data_path,
                         has_predictors=args.has_predictors)
        
        data, indices = self.data.numpy(return_idx=True)
        self.dataset = TimeSeriesDataset(data=data,indices=indices,
                                        window=args.window,
                                        horizon=args.horizon)
        
        train_split = int(len(self.dataset) * 0.4)
        val_split = int(len(self.dataset)  * 0.6)
        test_split = int(len(self.dataset)  * 0.8)
        # coarse indices by window size
        train_indices = np.array([i for i in range(train_split) if i % self.dataset.window == 0])
        val_indices = np.array([i for i in range(val_split) if i % self.dataset.window == 0])
        test_indices = np.array([i for i in range(test_split) if i % self.dataset.window == 0])
        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False)