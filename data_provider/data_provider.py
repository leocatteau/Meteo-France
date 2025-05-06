import numpy as np
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import bdclim, bdclim_clean
from data_provider.data_preparation import WindowHorizonDataset, SequenceMaskDataset, SampleMaskDataset


# these classes are for creating the compatible data objects for the models, depending on the training process

data_dict = {
    'bdclim': bdclim,
    'bdclim_clean': bdclim_clean
    # 'Ornstein_Uhlenbeck': Ornstein_Uhlenbeck_Data,
}
dataset_dict = {
    'SampleMaskDataset': SampleMaskDataset,
    'SequenceMaskDataset': SequenceMaskDataset,
    'WindowHorizonDataset': WindowHorizonDataset
}

class DataProvider:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size

        assert args.data in data_dict, f"Data {args.data} not found in data dictionary: {data_dict.keys()}"
        assert args.dataset in dataset_dict, f"Dataset {args.dataset} not found in dataset dictionary: {dataset_dict.keys()}"
        Data = data_dict[args.data]
        Dataset = dataset_dict[args.dataset]

        self.data = Data(root_path=args.root_path,data_path=args.data_path)
        self.dataset = Dataset(args=self.args, data_source=self.data)
        
        train_split = int(len(self.dataset) * 0.5)
        val_split = int(len(self.dataset)  * 0.1)
        test_split = int(len(self.dataset)  * 0.4)
        # coarse indices by window size
        train_indices = np.array([i for i in range(train_split) if i % self.dataset.coarse_frequency == 0])
        val_indices = np.array([i for i in train_indices[-1]+range(val_split) if i % self.dataset.coarse_frequency == 0])
        test_indices = np.array([i for i in val_indices[-1]+ range(test_split) if i % self.dataset.coarse_frequency == 0])
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False)