import numpy as np
import torch 
import xarray as xr
import os
import einops

from models.baseline import mean_fill
from trainer.custom_losses import masked_MSE

class Filler():
    def __init__(self, data_kwargs, model, model_kwargs, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root_path = data_kwargs.root_path
        self.data_path = data_kwargs.data_path
        self.ideal = data_kwargs.ideal if hasattr(data_kwargs, 'ideal') else False
        self.load_data()

        model_kwargs['seq_dim'] = self.clean_data.shape[1]
        self.model = model(**model_kwargs).to(self.device)
        self.load_model(args.model_path)
        self.mean_model = mean_fill(columnwise=True).to(self.device)

        self.corrupt = args.corrupt if hasattr(args, 'corrupt') else False
        self.mask_proba = args.mask_proba if hasattr(args, 'mask_proba') else 0.5
        self.mask_length = args.mask_length if hasattr(args, 'mask_length') else 24*7*3
        self.corrupt_data()

        self.window_size = args.window
        self.overlap = args.overlap if hasattr(args, 'overlap') else 0

    
    def load_data(self):
        # load dataset
        dataset = xr.load_dataset(os.path.join(self.root_path, self.data_path))
        # set dataset dataframe
        df = dataset.reset_coords()['t'].to_pandas()

        if self.ideal:
            # drop stations with NaN values
            df = df.dropna(axis=1, how='any')
        else:
            # drop stations with only NaN values
            df = df.dropna(axis=1, how='all')
            # # drop stations with less than 5 years data
            threshold = 5 * 365 * 24
            df = df.loc[:, df.count() >= threshold]
            print("remaining stations after 5 years threshold: ", df.shape[1])

        # load corrupted data
        self.clean_data = df.values
        self.clean_data = torch.tensor(self.clean_data, dtype=torch.float32).to(self.device)

        # set exogenous variables (predictors) dataframe
        self.predictors = dataset.reset_coords().drop_vars(['t','Station_Name','reseau_poste_actuel','lat','lon']).isel(time=0).to_dataframe().drop(columns='time')
        mask = (~np.isnan(df.values)).astype('uint8')
        self.mask = mask
        self.mask = torch.tensor(self.mask, dtype=torch.bool)

    def corrupt_data(self):
        if self.corrupt:
            # artificial masking
            eval_mask = torch.tensor(np.random.rand(len(self.clean_data), self.clean_data.shape[1]) > self.mask_proba/self.mask_length, dtype=torch.bool)
            masked_indices =  np.where(~eval_mask)
            for i,j in zip(masked_indices[0], masked_indices[1]):
                start = max(0, i - int(0.5 * self.mask_length))
                end = min(len(self.clean_data), i + int(0.5 * self.mask_length))
                eval_mask[start:end, j] = False
            self.eval_mask = eval_mask.clone()
            self.eval_mask[~self.mask] = True
            self.mask[~self.eval_mask] = False
        else:
            self.eval_mask = torch.ones_like(self.mask, dtype=torch.bool)

        self.corrupted_data = self.clean_data.clone()
        self.corrupted_data[~self.eval_mask] = torch.nan

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.eval()
    
    def predict(self, x, mask):
        # [b s n c]
        x_mean = self.mean_model(x,mask)
        prediction = self.model(x_mean, mask)
        return prediction

    def reconstruct(self):
        x_clean = self.clean_data
        x = self.corrupted_data
        m = self.mask
        eval_m = self.eval_mask

        # reshape the data to multiple batches of size window_size (abandon the len%window_size first steps)
        if x.shape[0] % self.window_size != 0:
            x_clean = x_clean[(x_clean.shape[0] % self.window_size):]
            x = x[(x.shape[0] % self.window_size):]
            m = m[(m.shape[0] % self.window_size):]
            eval_m = eval_m[(eval_m.shape[0] % self.window_size):]
            self.corrupted_data = self.corrupted_data[(self.corrupted_data.shape[0] % self.window_size):]
            self.mask = self.mask[(self.mask.shape[0] % self.window_size):]
            self.eval_mask = self.eval_mask[(self.eval_mask.shape[0] % self.window_size):]
        x_clean = einops.rearrange(x_clean, '(b w) n -> b w n 1', w=self.window_size)
        x = einops.rearrange(x, '(b w) n -> b w n 1', w=self.window_size)
        m = einops.rearrange(m, '(b w) n -> b w n 1', w=self.window_size)
        eval_m = einops.rearrange(eval_m, '(b w) n -> b w n 1', w=self.window_size)

        x_pred = torch.zeros_like(x)
        loss = 0.0
        for window in range(x.shape[0]):
            print(f'Processing window {window+1}/{x.shape[0]}')
            x_pred[window] = self.predict(x[window].unsqueeze(0), m[window].unsqueeze(0))
            loss += masked_MSE(x_pred[window], x_clean[window], eval_m[window])
        loss /= x.shape[0]

        # reshape the prediction to the original shape 
        x_pred = einops.rearrange(x_pred, 'b w n 1 -> (b w) n')
        self.reconstructed_data = x_pred.cpu().detach().numpy()

        return self.corrupted_data, self.predictors, self.mask, self.eval_mask, self.reconstructed_data, loss