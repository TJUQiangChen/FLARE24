from torch.utils.data import DataLoader
from .dataset_train import flare22_dataset
from .dataset_val import valp_dataset
from prefetch_generator import BackgroundGenerator
import torch
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_loader(config,data_size, data_path,unlab_data_path, pool_op_kernel_sizes, num_each_epoch):
    
    if config.DATASET.WITH_VAL:
        val_dataset = valp_dataset(config)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if config.DIS else None
        val_loader = DataLoaderX(
            dataset=val_dataset,
            sampler=val_sampler ,
            batch_size = config.DATALOADER.BATCH_SIZE, 
            num_workers=config.DATALOADER.NUM_WORKERS,
            pin_memory= config.DATALOADER.PIN_MEMORY, 
            shuffle=False,
            drop_last=False
        )
    else:
        val_loader = None
    
    
    train_dataset = flare22_dataset(config, data_size, data_path, unlab_data_path,  pool_op_kernel_sizes, num_each_epoch,is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size = config.DATALOADER.BATCH_SIZE, 
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory= config.DATALOADER.PIN_MEMORY, 
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    return train_loader,val_loader




