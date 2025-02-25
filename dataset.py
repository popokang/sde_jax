# import numpy as np
# import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils import data 
# from torchvision.datasets import MNIST
# import torchvision.datasets as datasets

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)



import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split



def get_dataset(config):
    resolution_size = config.data.image_size
    image_size = (resolution_size, resolution_size)
    batch_size = config.training.batch_size
    # num_workers = config.training.num_workers

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1 if config.data.centered else x)
        ])
    train_dataset = CIFAR10('./data', train=True, transform=train_transform, download=True)
    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1 if config.data.centered else x)
        ])
    valid_dataset = CIFAR10(root='./data', train=False, download=True, transform=valid_transform)


    train_loader = NumpyLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last = True)
    val_loader = NumpyLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last = True)


    return train_loader, val_loader, 

if __name__ == "__main__":
    from configs.cifar10_continuous import get_config
    config = get_config()
    train_data, test_data = get_dataset(config)
    x = next(iter(train_data))
    print(x[0].shape)
    print(x[0].dtype)
    print(x[0].min(), x[0].max()) 