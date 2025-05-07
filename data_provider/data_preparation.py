import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset

# these classes are the training preparation layers over the raw datasets, for liberty in masking processes, etc

class WindowHorizonDataset(Dataset):
    def __init__(self,args,data_source):
        super(WindowHorizonDataset, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        self.data, self.indices = data_source.pytorch(return_idx=True)
        self.data = torch.tensor(self.data)
        self.window = args.window
        self.horizon = args.horizon if args.horizon else 0
        self.scaler = args.scaler
        self.mask = torch.tensor(data_source.mask, dtype=torch.bool)
        self.mask_length = args.mask_length
        self.coarse_frequency = self.window + self.horizon if self.horizon>0 else self.window

        # artificial masking
        eval_mask = torch.tensor(np.random.rand(len(data_source), data_source.n_nodes) > args.mask_proba/self.mask_length, dtype=torch.bool)
        masked_indices =  np.where(~eval_mask)
        for i,j in zip(masked_indices[0], masked_indices[1]):
            start = max(0, i - int(0.5 * self.mask_length))
            end = min(len(data_source), i + int(0.5 * self.mask_length))
            eval_mask[start:end, j] = False
        self.eval_mask = eval_mask.clone()
        self.eval_mask[~self.mask] = True
        self.mask[~self.eval_mask] = False

        self.corrupted_data = self.data.clone()
        self.corrupted_data[~self.eval_mask] = torch.nan

    def __getitem__(self, index):
        sample = dict()
        sample['mask'] = self.mask[index:index+self.window].unsqueeze(-1).to(self.device)
        sample['eval_mask'] = self.eval_mask[index:index+self.window].unsqueeze(-1).to(self.device)

        sample['x'] = self.corrupted_data[index:index+self.window].unsqueeze(-1).to(self.device)

        if self.horizon > 0:
            raise NotImplementedError("out of sample not implemented yet.")
        else:
            sample['y'] = self.data[index:index+self.window].unsqueeze(-1).to(self.device)

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
        data = self.data
        return torch.FloatTensor(data)

    
class SequenceMaskDataset(Dataset):
    def __init__(self,args,data_source):
        super(SequenceMaskDataset, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.data = torch.tensor(self.data)
        self.mask_length = args.mask_length
        self.scaler = args.scaler
        self.mask = torch.tensor(data_source.mask, dtype=torch.bool)
        self.coarse_frequency = 1

        # artificial masking
        eval_mask = torch.tensor(np.random.rand(len(data_source), data_source.n_nodes) > args.mask_proba/self.mask_length, dtype=torch.bool)
        masked_indices =  np.where(~eval_mask)
        for i,j in zip(masked_indices[0], masked_indices[1]):
            start = max(0, i - int(0.5 * self.mask_length))
            end = min(len(data_source), i + int(0.5 * self.mask_length))
            eval_mask[start:end, j] = False
        self.eval_mask = eval_mask.clone()
        self.eval_mask[~self.mask] = True
        self.mask[~self.eval_mask] = False

        self.corrupted_data = self.data.clone()
        self.corrupted_data[~self.eval_mask] = torch.nan

    def __getitem__(self, index):
        sample = dict()
        sample['mask'] = self.mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['eval_mask'] = self.eval_mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)

        sample['x'] = self.corrupted_data[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['y'] = self.data[index].unsqueeze(0).unsqueeze(-1).to(self.device)

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
        data = self.data
        return torch.FloatTensor(data)
    

class SampleMaskDataset(Dataset):
    def __init__(self,args,data_source):
        super(SampleMaskDataset, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.data = torch.tensor(self.data)
        self.scaler = args.scaler
        self.mask = torch.tensor(data_source.mask, dtype=torch.bool)
        self.coarse_frequency = 1

        # artificial masking
        self.eval_mask = torch.zeros((len(data_source), data_source.n_nodes), dtype=torch.bool)
        self.eval_mask[torch.rand(len(data_source), data_source.n_nodes) > args.mask_proba] = True
        self.eval_mask[~self.mask] = True
        self.mask[~self.eval_mask] = False

        self.corrupted_data = self.data.clone()
        self.corrupted_data[~self.eval_mask] = torch.nan

    def __getitem__(self, index):
        sample = dict()
        sample['mask'] = self.mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['eval_mask'] = self.eval_mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)

        sample['x'] = self.corrupted_data[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['y'] = self.data[index].unsqueeze(0).unsqueeze(-1).to(self.device) 

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
        data = self.data
        return torch.FloatTensor(data)


class PatchMaskDataset(Dataset):
    def __init__(self,args,data_source):
        super(PatchMaskDataset, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        self.data, self.indices = data_source.numpy(return_idx=True)
        self.data = torch.tensor(self.data)
        self.scaler = args.scaler
        self.mask = torch.tensor(data_source.mask, dtype=torch.bool)
        self.coarse_frequency = 1

        # artificial masking
        corrupted_sations = np.random.choice(data_source.n_nodes, size=int(data_source.n_nodes * args.mask_proba), replace=False)
        self.eval_mask = torch.ones((len(data_source), data_source.n_nodes), dtype=torch.bool)
        self.eval_mask[int(len(data_source) * args.treshold_time):, corrupted_sations] = False
        self.eval_mask[~self.mask] = True
        self.mask[~self.eval_mask] = False

        self.corrupted_data = self.data.clone()
        self.corrupted_data[~self.eval_mask] = torch.nan

    def __getitem__(self, index):
        sample = dict()
        sample['mask'] = self.mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['eval_mask'] = self.eval_mask[index].unsqueeze(0).unsqueeze(-1).to(self.device)

        sample['x'] = self.corrupted_data[index].unsqueeze(0).unsqueeze(-1).to(self.device)
        sample['y'] = self.data[index].unsqueeze(0).unsqueeze(-1).to(self.device) 

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
        data = self.data
        return torch.FloatTensor(data)

