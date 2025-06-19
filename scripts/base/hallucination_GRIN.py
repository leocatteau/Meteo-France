import torch
import json

import sys
sys.path.append('..')

########################################################################
from data_provider.data_provider import DataProvider
from models.GRIN import GRINet
from trainer.Filler import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'PatchMaskDataset'
    data_kwargs.root_path = '../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2023-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 1
    data_kwargs.mask_length = 24*7*10
    data_kwargs.mask_proba = 0.95
    data_kwargs.treshold_time = 0.25
    data_kwargs.window = 24*7*1
    data_kwargs.horizon = 0

    data_provider = DataProvider(data_kwargs)
    adjacency_matrix = torch.FloatTensor(data_provider.data.umap_adjacency(threshold=0.0, verbose=False)).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = data_provider.dataloader()
    clean_data = data_provider.dataset.data
    eval_mask = data_provider.dataset.eval_mask

    model_kwargs = dict(adj=adjacency_matrix, d_in=1, d_ff=data_provider.data.n_nodes, global_att=True)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 1e-5
    filler_kwargs.epochs = 100
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Filler(GRINet, model_kwargs, filler_kwargs)
    filler.load_model('../trained_models/GRINet.pt')

    corrupted_data, reconstructed_data = filler.reconstruct_from_loader(dataloader, get_original_data=True)

    results = {
        'clean_data': clean_data.detach().cpu().numpy().tolist(),
        'eval_mask': eval_mask.detach().cpu().numpy().tolist(),
        'reconstructed_data': reconstructed_data.detach().cpu().numpy().tolist(),
        'predictors': data_provider.data.predictors.to_json()
    }

    with open(f'../../results/GRINet_hallucinated_bdclim_safran_2023-2024.nc.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()

