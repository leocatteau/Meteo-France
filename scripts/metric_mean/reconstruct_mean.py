import json
import pickle

import sys
sys.path.append('../..')

########################################################################
from models.mean import identity
from models.local_mean import local_mean
from training.inference import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2022-2024.nc'
    data_kwargs.ideal = True

    model_kwargs = dict()

    filler_kwargs = SimpleNamespace()
    filler_kwargs.mask_proba = 0.9
    filler_kwargs.mask_length = 24*7*3
    filler_kwargs.window = 24*1*1
    filler_kwargs.overlap = False

    filler = Filler(data_kwargs, local_mean, model_kwargs, filler_kwargs)
    original_data, corrupted_data, predictors, mask, eval_mask, reconstructed_data, RMSE, MAE, RG_RMSE, RG_MAE, GRAD_RMSE, GRAD_MAE = filler.reconstruct()

    losses = {
        'RMSE': RMSE,
        'MAE': MAE,
        'RG_RMSE': RG_RMSE,
        'RG_MAE': RG_MAE,
        'GRAD_RMSE': GRAD_RMSE,
        'GRAD_MAE': GRAD_MAE
    }

    print(f'RG RMSE:{RG_RMSE}')

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/metric_mean/data/reconstruction_losses_local_mean.pkl', 'wb') as file:
        pickle.dump(losses, file)
    print("Reconstruction saved.")

    with open(f'../../../results/metric_mean/data/reconstructed_data_local_mean.pkl', 'wb') as file:
        pickle.dump(reconstructed_data.tolist(), file)

    with open(f'../../../results/metric_mean/data/original_data_local_mean.pkl', 'wb') as file:
        pickle.dump(original_data.tolist(), file)

    with open(f'../../../results/metric_mean/data/corrupted_data_local_mean.pkl', 'wb') as file:
        pickle.dump(corrupted_data.tolist(), file)

    with open(f'../../../results/metric_mean/data/mask_local_mean.pkl', 'wb') as file:
        pickle.dump(mask.tolist(), file)

    with open(f'../../../results/metric_mean/data/predictors_local_mean.json', 'w') as file:
        json.dump(predictors.to_json(), file)

    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

