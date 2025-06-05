import json

import sys
sys.path.append('../..')

########################################################################
from models.MLP import MLP
from trainer.Filler import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2020-2024.nc'
    data_kwargs.ideal = False

    model_kwargs = dict(hidden_dim=512)

    filler_kwargs = SimpleNamespace()
    filler_kwargs.window = 24*1*1
    filler_kwargs.model_path = '../../../results/DoubleMask_MLP/model/MLP_100.pt'

    filler = Filler(data_kwargs, MLP, model_kwargs, filler_kwargs)
    corrupted_data, predictors, mask, reconstructed_data = filler.reconstruct()

    results = {
        'original_data': corrupted_data.tolist(),
        'predictors': predictors.to_json(),
        'mask': mask.tolist(),
        'reconstructed_data': reconstructed_data.tolist()
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/DoubleMask_MLP/reconstruction_100.json', 'w') as file:
        json.dump(results, file, indent=4)
    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

