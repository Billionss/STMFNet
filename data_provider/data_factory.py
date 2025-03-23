from .data_loader import MyDataset
from .formers_loader import Dataset_Custom
from torch.utils.data import DataLoader
from torchvision import transforms

# data_dict = {
#     'Beijing': Dataset_Custom,
#     'Tianjin': Dataset_Custom,
# }

data_dict = {
    'Beijing': MyDataset,
    'Tianjin': MyDataset,
}

to_tensor = transforms.ToTensor()


# flag = 'train' or 'val' or 'test'
def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    else: # train or val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        dataset=args.data,
        flag=flag,
        window_size=args.seq_len,
        horizon=args.label_len,
        transform=None,
        image_transform=to_tensor,
    )
    
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader
