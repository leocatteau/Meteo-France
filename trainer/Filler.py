import numpy as np
import torch 
import xarray as xr
import os
import einops

from models.baseline import mean_fill

class Filler():
    def __init__(self, data_kwargs, model, model_kwargs, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root_path = data_kwargs.root_path
        self.data_path = data_kwargs.data_path
        self.ideal = data_kwargs.ideal if hasattr(data_kwargs, 'ideal') else False
        self.load_data()

        model_kwargs['seq_dim'] = self.corrupted_data.shape[1]
        self.model = model(**model_kwargs).to(self.device)
        self.load_model(args.model_path)
        self.mean_model = mean_fill(columnwise=True).to(self.device)

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
            # drop columns with more than 90% NaN values
            threshold = 1 * df.shape[0]
            df = df.dropna(axis=1, thresh=threshold)
            print(f"Remaining stations: {df.shape[1]} out of {dataset.sizes['num_poste']}")

        # load corrupted data
        self.corrupted_data = df.values

        # set exogenous variables (predictors) dataframe
        self.predictors = dataset.reset_coords().drop_vars(['t','Station_Name','reseau_poste_actuel','lat','lon']).isel(time=0).to_dataframe().drop(columns='time')
        mask = (~np.isnan(df.values)).astype('uint8')
        self.mask = mask

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.eval()
    
    def predict(self, x, mask):
        # [b s n c]
        x_mean = self.mean_model(x,mask)
        prediction = self.model(x_mean, mask)
        return prediction

    def reconstruct(self):
        x = torch.tensor(self.corrupted_data, dtype=torch.float32).to(self.device)
        m = torch.tensor(self.mask, dtype=torch.bool).to(self.device)

        # reshape the data to multiple batches of size window_size (abandon the len%window_size first steps)
        if x.shape[0] % self.window_size != 0:
            x = x[(x.shape[0] % self.window_size):]
            m = m[(m.shape[0] % self.window_size):]
            self.corrupted_data = self.corrupted_data[(self.corrupted_data.shape[0] % self.window_size):]
            self.mask = self.mask[(self.mask.shape[0] % self.window_size):]
        x = einops.rearrange(x, '(b w) n -> b w n 1', w=self.window_size)
        m = einops.rearrange(m, '(b w) n -> b w n 1', w=self.window_size)

        x_pred = torch.zeros_like(x)
        for window in range(x.shape[0]):
            print(f'Processing window {window+1}/{x.shape[0]}')
            x_pred[window] = self.predict(x[window].unsqueeze(0), m[window].unsqueeze(0))

        # reshape the prediction to the original shape 
        x_pred = einops.rearrange(x_pred, 'b w n 1 -> (b w) n')
        self.reconstructed_data = x_pred.cpu().detach().numpy()

        return self.corrupted_data, self.predictors, self.mask, self.reconstructed_data