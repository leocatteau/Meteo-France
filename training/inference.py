import numpy as np
import torch 
import xarray as xr
import os
import einops
# import umap
from sklearn.neighbors import kneighbors_graph

from models.baseline import mean_fill
from models.mean import identity
from training.custom_losses import masked_MSE, masked_MAE, masked_RMSE, masked_RG_RMSE_MAE, temporal_gradient_RMSE, temporal_gradient_MAE


class Filler():
    def __init__(self, data_kwargs, model, model_kwargs, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.root_path = data_kwargs.root_path
        self.data_path = data_kwargs.data_path
        self.corrupted_data_path = data_kwargs.corrupted_data_path if hasattr(data_kwargs, 'corrupted_data_path') else None
        self.ideal = data_kwargs.ideal if hasattr(data_kwargs, 'ideal') else False
        self.load_data()

        # model_kwargs['seq_dim'] = self.original_data.shape[1]
        model_kwargs['adj'] = self.adjacency_matrix
        # model_kwargs['predictors'] = self.predictors
        self.model = model(**model_kwargs).to(self.device)
        self.load_model(args.model_path)
        # self.model = identity().to(self.device)
        self.mean_model = mean_fill(columnwise=True).to(self.device)

        self.mask_proba = args.mask_proba if hasattr(args, 'mask_proba') else 0.5
        self.mask_length = args.mask_length if hasattr(args, 'mask_length') else 24*7*3
        self.corrupt_data()

        self.window_size = args.window
        self.overlap = args.overlap if hasattr(args, 'overlap') else 0

        self.mask = self.mask.to(self.device)
        self.corrupted_data = self.corrupted_data.to(self.device)

        
    def load_data(self):
        # load dataset
        dataset = xr.load_dataset(os.path.join(self.root_path, self.data_path))

        if self.ideal:
            # drop stations with NaN values
            dataset = dataset.dropna(dim='num_poste', how='any')
        else:
            # drop stations with only NaN values
            dataset = dataset.dropna(dim='num_poste', how='all')
            # # drop stations with less than 5 years data
            # valid_stations = (dataset['t'].count(dim='time') >= 5 * 365 * 24)
            # dataset = dataset.isel(num_poste=valid_stations)
            # print("remaining stations after 5 years threshold: ", dataset.dims['num_poste'])

        # load original data
        self.original_data = dataset['t'].values
        self.original_data = torch.tensor(self.original_data, dtype=torch.float32)

        # set exogenous variables (predictors) dataframe
        self.predictors = dataset.reset_coords().drop_vars(['t','Station_Name','reseau_poste_actuel']).isel(time=0).to_dataframe().drop(columns='time')
        mask = (~np.isnan(dataset['t'].values)).astype('uint8')
        self.mask = mask
        self.mask = torch.tensor(self.mask, dtype=torch.bool)

        # compute adjacency matrix
        #adjacency_matrix = self.correlation_adjacency(threshold=0.9)
        adjacency_matrix = self.KNN_adjacency(threshold=0.0)
        #adjacency_matrix = self.umap_adjacency(threshold=0.0)
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

    def correlation_adjacency(self, threshold=0.1):
        corr_matrix = self.df.corr()
        corr_matrix[corr_matrix < threshold] = 0
        corr_matrix.fillna(0, inplace=True)
        corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))
        return corr_matrix.values
    
    # def umap_adjacency(self, threshold=0.1):
    #     # select only labx lamby and ZS as predictors
    #     predictors = self.predictors[['lambx', 'lamby', 'ZS']]
    #     predictors = (predictors - predictors.mean()) / predictors.std()
    #     reducer = umap.UMAP(min_dist=0.5, n_neighbors=10, metric='euclidean')
    #     reducer.fit_transform(predictors.fillna(method='ffill'))

    #     adjacency_matrix = reducer.graph_.toarray()
    #     adjacency_matrix[adjacency_matrix < threshold] = 0
    #     adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    #     return adjacency_matrix
    
    def KNN_adjacency(self, threshold=0.1):
        # select only labx lamby and ZS as predictors
        # predictors = self.predictors[['lambx', 'lamby', 'ZS']]
        # predictors = (predictors - predictors.mean()) / predictors.std()
        predictors = self.predictors[['lambx', 'lamby', 'ZS', 'TCD_500m', 'TPI_30m', 'aspect_30m', 'slope_30m', 'slope_500m', 'aspect_500m', 'TPI_1000m']]
        predictors = (predictors - predictors.mean()) / predictors.std()

        adjacency_matrix = kneighbors_graph(predictors.fillna(method='ffill'), n_neighbors=10, mode='connectivity', include_self=False).toarray()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
        return adjacency_matrix

    def corrupt_data(self):
        if self.corrupted_data_path is None:
            # artificial masking
            eval_mask = torch.tensor(np.random.rand(len(self.original_data), self.original_data.shape[1]) > self.mask_proba/self.mask_length, dtype=torch.bool)
            masked_indices =  np.where(~eval_mask)
            for i,j in zip(masked_indices[0], masked_indices[1]):
                start = max(0, i - int(0.5 * self.mask_length))
                end = min(len(self.original_data), i + int(0.5 * self.mask_length))
                eval_mask[start:end, j] = False
            self.eval_mask = eval_mask.clone()
            self.eval_mask[~self.mask] = True
            self.mask[~self.eval_mask] = False

            self.corrupted_data = self.original_data.clone()
            self.corrupted_data[~self.eval_mask] = torch.nan
        else:
            # load dataset
            dataset = xr.load_dataset(os.path.join(self.root_path, self.corrupted_data_path))
            # set dataset dataframe
            df = dataset.reset_coords()['t'].to_pandas()

            if self.ideal:
                # drop stations with NaN values
                df = df.dropna(axis=1, how='any')
            else:
                # drop stations with only NaN values
                df = df.dropna(axis=1, how='all')
                # # drop stations with less than 5 years data
                # threshold = 5 * 365 * 24
                # df = df.loc[:, df.count() >= threshold]
                # print("remaining stations after 5 years threshold: ", df.shape[1])

            # load corrupted data
            self.corrupted_data = df.values
            self.corrupted_data = torch.tensor(self.corrupted_data, dtype=torch.float32).to(self.device)

            # set exogenous variables (predictors) dataframe
            # mask = np.array(~np.isnan(self.original_data)).astype('uint8')
            eval_mask = (~np.isnan(df.values)).astype('uint8')
            eval_mask[~self.mask] = True
            self.eval_mask = torch.tensor(eval_mask, dtype=torch.bool)
            self.mask[~self.eval_mask] = False
            # mask = (~np.isnan(df.values)).astype('uint8')
            # self.mask = torch.tensor(mask, dtype=torch.bool)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
    
    def predict(self, x, mask):
        # [b s n c]
        x_mean = self.mean_model(x,mask)
        prediction = self.model(x_mean, mask)
        return prediction

    # def reconstruct(self):
    #     self.model.eval()
    #     x_clean = self.original_data
    #     x = self.corrupted_data
    #     m = self.mask
    #     eval_m = self.eval_mask

    #     # reshape the data to multiple batches of size window_size (abandon the len%window_size first steps)
    #     if x.shape[0] % self.window_size != 0:
    #         x_clean = x_clean[(x_clean.shape[0] % self.window_size):]
    #         x = x[(x.shape[0] % self.window_size):]
    #         m = m[(m.shape[0] % self.window_size):]
    #         eval_m = eval_m[(eval_m.shape[0] % self.window_size):]
    #         self.corrupted_data = self.corrupted_data[(self.corrupted_data.shape[0] % self.window_size):]
    #         self.mask = self.mask[(self.mask.shape[0] % self.window_size):]
    #         self.eval_mask = self.eval_mask[(self.eval_mask.shape[0] % self.window_size):]
    #     x_clean = einops.rearrange(x_clean, '(b w) n -> b w n 1', w=self.window_size)
    #     x = einops.rearrange(x, '(b w) n -> b w n 1', w=self.window_size)
    #     m = einops.rearrange(m, '(b w) n -> b w n 1', w=self.window_size)
    #     eval_m = einops.rearrange(eval_m, '(b w) n -> b w n 1', w=self.window_size)
    #     avoid_m = ((eval_m)^(~m.cpu()))

    #     x_pred = torch.zeros_like(x).cpu()
    #     RMSE = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
    #     MAE = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
    #     RG_RMSE = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
    #     RG_MAE = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
    #     with torch.no_grad():
    #         for window in range(x.shape[0]):
    #             print(f'Processing window {window+1}/{x.shape[0]}')

    #             if self.overlap:
    #                 x_pred_gpu = self.predict(torch.cat([x[-window+1], x[-window]], dim=0).unsqueeze(0), torch.cat([m[-window+1], m[-window]], dim=0).unsqueeze(0))[:, :self.window_size, :, :]
    #                 x_pred[-window] = x_pred_gpu.cpu()
    #             else:
    #                 x_pred_gpu = self.predict(x[-window].unsqueeze(0), m[-window].unsqueeze(0))
    #                 x_pred[-window] = x_pred_gpu.cpu()
    #             print(x_pred[-window].shape)
    #             print(avoid_m[-window].shape)
    #             RMSE[-window] = masked_RMSE(x_pred[-window], x_clean[-window], eval_m[-window])
    #             MAE[-window] = masked_MAE(x_pred[-window], x_clean[-window], eval_m[-window])
    #             RG_RMSE[-window], RG_MAE[-window] = masked_RG_RMSE_MAE(x_pred[-window], x_clean[-window], self.adjacency_matrix, avoid_m[-window])

    #     # reshape the prediction to the original shape 
    #     x_pred = einops.rearrange(x_pred, 'b w n 1 -> (b w) n')
    #     self.reconstructed_data = x_pred.cpu().detach().numpy()

    #     return self.original_data, self.corrupted_data, self.predictors, self.mask, self.eval_mask, self.reconstructed_data, RMSE, MAE, RG_RMSE, RG_MAE

    # def reconstruct(self):
    #     self.model.eval()
    #     valid_length = self.corrupted_data.shape[0] - self.corrupted_data.shape[0] % self.window_size
    #     original_data = self.original_data[-valid_length:]
    #     corrupted_data = self.corrupted_data[-valid_length:]
    #     mask_tensor = self.mask[-valid_length:]
    #     evaluation_mask = self.eval_mask[-valid_length:]

    #     batch_count = valid_length // self.window_size
    #     input_data = einops.rearrange(corrupted_data, '(b w) n -> b w n 1', b=batch_count, w=self.window_size)
    #     input_mask = einops.rearrange(mask_tensor, '(b w) n -> b w n 1', b=batch_count, w=self.window_size)

    #     with torch.no_grad():
    #         prediction_gpu = self.predict(input_data, input_mask)
    #     prediction = einops.rearrange(prediction_gpu, 'b w n 1 -> (b w) n 1').cpu()
    #     original_data = einops.rearrange(original_data, 'w n -> w n 1').cpu()
    #     evaluation_mask = einops.rearrange(evaluation_mask, 'w n -> w n 1').cpu()
    #     mask_tensor = einops.rearrange(mask_tensor, 'w n -> w n 1').cpu()

    #     rmse_value = masked_RMSE(prediction, original_data.cpu(), evaluation_mask.cpu())
    #     mae_value = masked_MAE(prediction, original_data.cpu(), evaluation_mask.cpu())
    #     rg_rmse_value, rg_mae_value = masked_RG_RMSE_MAE(prediction, original_data.cpu(), self.adjacency_matrix, (~evaluation_mask | mask_tensor.cpu()))
    #     rmse_grad = temporal_gradient_RMSE(original_data, prediction, evaluation_mask)
    #     mae_grad = temporal_gradient_MAE(original_data, prediction, evaluation_mask)

    #     prediction = einops.rearrange(prediction, 'w n 1 -> w n')
    #     original_data = einops.rearrange(original_data, 'w n 1 -> w n').cpu()
    #     evaluation_mask = einops.rearrange(evaluation_mask, 'w n 1 -> w n').cpu()
    #     mask_tensor = einops.rearrange(mask_tensor, 'w n 1 -> w n').cpu()

    #     self.reconstructed_data = prediction.numpy()
    #     return original_data.cpu().numpy(), corrupted_data.cpu().numpy(), mask_tensor.cpu().numpy(), evaluation_mask.cpu().numpy(), self.reconstructed_data, rmse_value.item(), mae_value.item(), rg_rmse_value.item(), rg_mae_value.item(), rmse_grad.item(), rmse_grad.item()

    def reconstruct(self):
        self.model.eval()
        stride_size = self.window_size - self.overlap
        sequence_length = self.corrupted_data.shape[0]
        usable_length = sequence_length - ((sequence_length - self.window_size) % stride_size)
        original_data = self.original_data[:usable_length]
        reconstruction_gpu = self.corrupted_data[:usable_length].clone()
        mask_gpu = self.mask[:usable_length].clone()
        evaluation_mask = self.eval_mask[:usable_length]

        last_start = usable_length - self.window_size
        for start_index in range(last_start, -1, -stride_size):
            end_index = start_index + self.window_size
            print(f'Processing window [{start_index},{end_index}] of {usable_length} indices')
            input_data = reconstruction_gpu[start_index:end_index].unsqueeze(-1).unsqueeze(0)
            input_mask = mask_gpu[start_index:end_index].unsqueeze(-1).unsqueeze(0)
            with torch.no_grad():
                prediction_window_gpu = self.predict(input_data, input_mask).squeeze(-1).squeeze(0)
            unfilled_gpu = ~mask_gpu[start_index:end_index]
            reconstruction_gpu[start_index:end_index][unfilled_gpu] = prediction_window_gpu[unfilled_gpu]
            mask_gpu[start_index:end_index][unfilled_gpu] = True

        reconstruction_cpu = einops.rearrange(reconstruction_gpu, 'w n -> w n 1').cpu()
        original_data = einops.rearrange(original_data, 'w n -> w n 1').cpu()
        evaluation_mask = einops.rearrange(evaluation_mask, 'w n -> w n 1').cpu()
        mask = einops.rearrange(mask_gpu, 'w n -> w n 1').cpu()

        rmse_value = masked_RMSE(reconstruction_cpu, original_data, evaluation_mask)
        mae_value = masked_MAE(reconstruction_cpu, original_data, evaluation_mask)
        rmse_grad = temporal_gradient_RMSE(original_data, reconstruction_cpu, evaluation_mask)
        mae_grad = temporal_gradient_MAE(original_data, reconstruction_cpu, evaluation_mask)

        rg_rmse_value, rg_mae_value = masked_RG_RMSE_MAE(
            reconstruction_cpu,
            original_data,
            self.adjacency_matrix,
            (~evaluation_mask | mask),
        )

        reconstruction_cpu = einops.rearrange(reconstruction_cpu, 'w n 1 -> w n')
        original_data = einops.rearrange(original_data, 'w n 1 -> w n').cpu()
        evaluation_mask = einops.rearrange(evaluation_mask, 'w n 1 -> w n').cpu()
        mask = einops.rearrange(mask, 'w n 1 -> w n').cpu()

        print(rmse_value.item())
        print(mae_value.item())
        print(rg_rmse_value.item())
        print(rg_mae_value.item())
        print(rmse_grad.item())
        print(mae_grad.item())

        print(original_data.shape)
        print(self.corrupted_data[:usable_length].shape)
        print(evaluation_mask.shape)
        print(reconstruction_cpu.shape)
        print(mask.shape)
        print(original_data.shape)

        return (
            self.predictors,
            original_data.numpy(),
            self.corrupted_data[:usable_length].cpu().numpy(),
            mask.numpy(),
            evaluation_mask.numpy(),
            reconstruction_cpu.numpy(),
            rmse_value.item(),
            mae_value.item(),
            rg_rmse_value.item(),
            rg_mae_value.item(),
            rmse_grad.item(),
            mae_grad.item()
        )