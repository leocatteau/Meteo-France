import json

import sys
sys.path.append('../..')

########################################################################
from models.mean import identity
from training.inference import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2022-2024.nc'
    data_kwargs.ideal = True

    model_kwargs = dict()

    filler_kwargs = SimpleNamespace()
    filler_kwargs.mask_proba = 0.5
    filler_kwargs.mask_length = 24*7*3
    filler_kwargs.window = 24*1*1
    filler_kwargs.overlap = False

    filler = Filler(data_kwargs, identity, model_kwargs, filler_kwargs)
    original_data, corrupted_data, predictors, mask, eval_mask, reconstructed_data, RMSE, MAE, RG_RMSE, RG_MAE = filler.reconstruct()

    results = {
        'original_data': original_data.tolist(),
        'corrupted_data': corrupted_data.tolist(),
        'predictors': predictors.to_json(),
        'mask': mask.tolist(),
        'reconstructed_data': reconstructed_data.tolist(),
        'RMSE': RMSE.tolist(),
        'MAE': MAE.tolist(),
        'RG_RMSE': RG_RMSE.tolist(),
        'RG_MAE': RG_MAE.tolist(),
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/metric_mean/data/reconstruction_diffusion_mean.json', 'w') as file:
        json.dump(results, file, indent=4)
    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

