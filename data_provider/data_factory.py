from data_provider.data_loader import bdclim_clean_mask, bdclim_clean_sequence_mask #, Ornstein_Uhlenbeck_Data
from torch.utils.data import DataLoader

data_dict = {
    'bdclim_clean_mask': bdclim_clean_mask,
    'bdclim_clean_sequence_mask': bdclim_clean_sequence_mask,
    # 'Ornstein_Uhlenbeck': Ornstein_Uhlenbeck_Data,
    # more to be added
}

def data_provider(args, flag): 
    Data = data_dict[args.data]

    shuffle = False if (flag == 'test') else True
    batch_size = args.batch_size

    dataset = Data(args = args,root_path=args.root_path,data_path=args.data_path, scale=args.scale, flag=flag)
    # print(flag)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataset, dataloader