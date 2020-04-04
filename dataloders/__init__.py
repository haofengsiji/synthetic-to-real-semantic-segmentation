from dataloders.datasets import gtav2cityscapes
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'gtav2cityscapes':
        train_set = gtav2cityscapes.TrainSet(args)
        val_set = gtav2cityscapes.ValSet(args)
        test_set = gtav2cityscapes.TestSet(args)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

