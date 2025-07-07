import json
import pickle

import sys
sys.path.append('../..')

########################################################################
from models.GRIN import GRINet
from training.inference import Filler 

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2022-2024.nc'
    data_kwargs.window = 24*1*1
    data_kwargs.ideal = True

    model_kwargs = dict(d_in=1, global_att=True, d_hidden_spatial=64, d_hidden_temporal=data_kwargs.window)

    filler_kwargs = SimpleNamespace()
    filler_kwargs.mask_proba = 0.9
    filler_kwargs.mask_length = 24*7*3
    filler_kwargs.window = 24*1*1
    filler_kwargs.overlap = False
    filler_kwargs.model_path = '../../../results/metric_GRIN/model/GRIN_24h.pt'

    filler = Filler(data_kwargs, GRINet, model_kwargs, filler_kwargs)
    original_data, corrupted_data, predictors, mask, eval_mask, reconstructed_data, RMSE, MAE, RG_RMSE, RG_MAE = filler.reconstruct()

    losses = {
        'RMSE': RMSE.tolist(),
        'MAE': MAE.tolist(),
        'RG_RMSE': RG_RMSE.tolist(),
        'RG_MAE': RG_MAE.tolist(),
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/metric_GRIN/data/reconstruction_losses_GRIN.pkl', 'wb') as file:
        pickle.dump(losses, file)
    print("Reconstruction saved.")

    with open(f'../../../results/metric_GRIN/data/reconstructed_data_GRIN.pkl', 'wb') as file:
        pickle.dump(reconstructed_data.tolist(), file)

    with open(f'../../../results/metric_GRIN/data/original_data_GRIN.pkl', 'wb') as file:
        pickle.dump(original_data.tolist(), file)

    with open(f'../../../results/metric_GRIN/data/corrupted_data_GRIN.pkl', 'wb') as file:
        pickle.dump(corrupted_data.tolist(), file)

    with open(f'../../../results/metric_GRIN/data/mask_GRIN.pkl', 'wb') as file:
        pickle.dump(mask.tolist(), file)

    with open(f'../../../results/metric_GRIN/data/predictors.json', 'w') as file:
        json.dump(predictors.to_json(), file)

    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

