import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.functions import Ornstein_Uhlenbeck

class bdclim_clean_mask(Dataset):
    def __init__(self, args, root_path, flag='train', data_path='bdclim_safran_2022-2024.nc', scale=False):
        self.args = args
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        data = xr.load_dataarray(os.path.join(self.root_path,self.data_path))
        train_data_array = data.dropna(dim='num_poste')
        X= train_data_array.values
        self.seq_len = X.shape[0]
        self.seq_dim = X.shape[1]
        # print('seq_len:', self.seq_len)
        # print('seq_dim:', self.seq_dim)

        num_train = int(self.seq_len * 0.7)
        num_test = int(self.seq_len * 0.2)
        num_vali = self.seq_len - num_train - num_test
        border1s = [0, num_train, num_train + num_test]
        border2s = [num_train, num_train + num_test, num_train + num_test + num_vali]
        border1 = border1s[self.flag]
        border2 = border2s[self.flag]
        
        mask = np.zeros((self.seq_len, self.seq_dim), dtype=bool)
        mask[np.random.rand(self.seq_len, self.seq_dim) < self.args.mask_proba] = 1
        data_X = X.copy()
        data_X[mask] = np.NaN
        data_y = X.copy()

        # self.data_X = data_X[border1:border2]
        # self.data_y = data_y[border1:border2]
        self.data_X = torch.tensor(data_X[border1:border2]).float()
        self.data_y = torch.tensor(data_y[border1:border2]).float()
        #self.mask = mask[border1:border2] should we pass the mask? why don't they in sequence setting ?

    def __getitem__(self, index):
        sample = (self.data_X[index], self.data_y[index])

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(sample[0])
            sample = (self.scaler.transform(sample[0]), sample[1])
        return sample
    
    def __len__(self):
        return self.seq_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

    
class bdclim_clean_sequence_mask(Dataset):
    def __init__(self, args, root_path, flag='train', data_path='bdclim_safran_2022-2024.nc', scale=False):
        self.args = args
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        data = xr.load_dataarray(os.path.join(self.root_path,self.data_path))
        train_data_array = data.dropna(dim='num_poste')
        X= train_data_array.values
        self.seq_len = X.shape[0]
        self.seq_dim = X.shape[1]
        # print('seq_len:', self.seq_len)
        # print('seq_dim:', self.seq_dim)

        num_train = int(self.seq_len * 0.7)
        num_test = int(self.seq_len * 0.2)
        num_vali = self.seq_len - num_train - num_test
        border1s = [0, num_train, num_train + num_test]
        border2s = [num_train, num_train + num_test, num_train + num_test + num_vali]
        border1 = border1s[self.flag]
        border2 = border2s[self.flag]
        
        # mask = np.ones((self.seq_len, self.seq_dim), dtype=bool)
        # mask[np.random.rand(self.seq_len, self.seq_dim) < self.args.mask_proba] = 0
        mask = np.array(np.random.rand(self.seq_len, self.seq_dim) < self.args.mask_proba/self.args.mask_len, dtype=bool)
        indices =  np.where(mask)
        for i,j in zip(indices[0], indices[1]):
            start = max(0, i - int(0.5 * self.args.mask_len))
            end = min(self.seq_len, i + int(0.5 * self.args.mask_len))
            mask[start:end, j] = True
        data_X = X.copy()
        data_X[mask] = np.NaN
        data_y = X.copy()

        # self.data_X = data_X[border1:border2]
        # self.data_y = data_y[border1:border2]
        self.data_X = torch.tensor(data_X[border1:border2]).float()
        self.data_y = torch.tensor(data_y[border1:border2]).float()
        #self.mask = mask[border1:border2] should we pass the mask? why don't they in sequence setting ?

    def __getitem__(self, index):
        sample = (self.data_X[index], self.data_y[index])

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(sample[0])
            sample = (self.scaler.transform(sample[0]), sample[1])
        return sample
    
    def __len__(self):
        return [int(self.seq_len * 0.7), int(self.seq_len * 0.2), int(self.seq_len * 0.1)][self.flag]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


# class Bdclim_matrix_sequence(Dataset):
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
#         self.scaler = StandardScaler()
#         data = xr.load_dataarray(os.path.join(self.root_path,self.data_path))
#         train_data_array = data.dropna(dim='num_poste')
#         X= train_data_array.values.T
#         self.seq_len = X.shape[0]
#         self.seq_dim = X.shape[1]

#         num_train = int(self.seq_len * 0.7)
#         num_test = int(self.seq_len * 0.2)
#         num_vali = self.seq_len - num_train - num_test
#         border1s = [0, num_train, num_train + num_test]
#         border2s = [num_train, num_train + num_test, num_train + num_test + num_vali]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.scale:
#             train_data = X[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(X)
#         else:
#             data = X

#         self.data = data[border1:border2]

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)



# class Ornstein_Uhlenbeck_Data(Dataset):

#     # transform args into a dict 
#     def __init__(self, d, N, dt, X0, mu, theta, train=True, scale=False):
#         """
#         Args:
#             d: dimension of the data (number of walkers)
#             N: number of time steps
#             dt: time step size
#             X0: initial position \in R^d
#             mu: long term mean per walker \in R^d
#             theta: mean reversion rate per walker \in R^d
#             train (bool, optional): If set to `True` load training data.
#             scale (bool, optional): Optional sclaing to be applied on a sample.
#         """

#         # data points
#         Y, sigma = Ornstein_Uhlenbeck(d, N, dt, X0, mu, theta)
#         self.sigma = sigma

#         # Set training and test data size
#         train_size=int(0.8*N)
#         self.train=train

#         if self.train:
#             X=X[:train_size]
#             Y=Y[:train_size]
#         else:
#             X=X[train_size:]
#             Y=Y[train_size:]

#         self.scale = scale

#         self.data=(X.astype(np.float32),Y.astype(np.float32))

#     def __len__(self):
#         return len(self.data[1])

#     def __getitem__(self, idx):

#         sample=(self.data[0][idx,...],self.data[1][idx])

#         if self.scale:
#             self.scaler = StandardScaler()
#             self.scaler.fit(sample[0])
#             sample = (self.scaler.transform(sample[0]), sample[1])

#         return sample
    
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    

# class Dataset_Custom(Dataset):
#     def __init__(self, args, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         self.args = args
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]

#         if self.set_type == 0 and self.args.augmentation_ratio > 0:
#             self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
