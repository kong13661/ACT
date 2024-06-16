from typing import Any
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os.path as osp
import os
import pytorch_lightning as pl
import torchvision
import pathlib
from PIL import Image
from convert_imagenet_lmdb import loads_pyarrow
import torch
import lmdb


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, train=True):
        db_path = str(db_path)
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        img, target = unpacked
        target = target - 1

        # imgbuf = unpacked
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return self.length
        else:
            return 50000

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class ImageNet64(pl.LightningDataModule):
    def __init__(self, batch_size, project_path):
        super().__init__()
        self.batch_size = batch_size
        self.project_path = pathlib.Path(project_path) / "Imagenet64" / "lmdb" / "lmdb"
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])

    def setup(self, stage):
        self.imagenet64_train = ImageFolderLMDB(
            self.project_path, transform = self.transform)
        self.imagenet64_val = ImageFolderLMDB(
            self.project_path, transform = self.transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.imagenet64_train, batch_size = self.batch_size, shuffle = True, num_workers=4, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.imagenet64_val, batch_size = self.batch_size, shuffle = False, num_workers=4, prefetch_factor=2)


class CIFAR10(pl.LightningDataModule):

    def __init__(self, batch_size, project_path):
        super().__init__()
        self.batch_size = batch_size
        self.project_path = project_path
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))])

        self.transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                               #    torchvision.transforms.RandomCrop(32, padding = 4),
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                (0.5, 0.5, 0.5))])

    def setup(self, stage):
        self.cifar10_train = datasets.CIFAR10(
            os.path.join(self.project_path, "pytorch"), train = True, download = True, transform = self.transform_train)
        self.cifar10_validation = datasets.CIFAR10(
            os.path.join(self.project_path, "pytorch"), train = True, download = True, transform = self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_validation, batch_size = self.batch_size, shuffle = False)


def get_data_module(config):
    if config['dataset'] == 'cifar10':
        return CIFAR10(config['batch_size'] // config['device'], config['path_auxiliary'].dataset), 3, 10
    if config['dataset'] == 'imagenet64':
        return ImageNet64(config['batch_size'] // config['device'], config['path_auxiliary'].dataset), 3, 10
