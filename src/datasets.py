import numpy as np
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch.utils.data import DataLoader
from torchvision import transforms

import ffcv_utils
from decoders_and_transforms import IMAGE_DECODERS

TRAIN_TRANSFORMS = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
])

TEST_TRANSFORMS = lambda size: transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
])


DS_TO_MEAN = {
    "IMAGENET": np.array([0.485, 0.456, 0.406]),
    "CALTECH101": np.array([0.0, 0.0, 0.0]),
    "CALTECH256": np.array([0.0, 0.0, 0.0]),
    "CIFAR10_0.1": np.array([0.4914, 0.4822, 0.4465]),
    "CIFAR10_0.25": np.array([0.4914, 0.4822, 0.4465]),
    "CIFAR10_0.5": np.array([0.4914, 0.4822, 0.4465]),
    "CIFAR10": np.array([0.4914, 0.4822, 0.4465]),
    "CIFAR100": np.array([0.5071, 0.4867, 0.4408]),
    "CHESTXRAY14": np.array([0.485, 0.456, 0.406]),
    "SUN397": np.array([0.0, 0.0, 0.0]),
    "AIRCRAFT": np.array([0.0, 0.0, 0.0]),
    "BIRDSNAP": np.array([0.0, 0.0, 0.0]),
    "FLOWERS": np.array([0.0, 0.0, 0.0]),
    "FOOD": np.array([0.0, 0.0, 0.0]),
    "PETS": np.array([0.0, 0.0, 0.0]),
    "STANFORD_CARS": np.array([0.0, 0.0, 0.0]),
}

DS_TO_STD = {
    "IMAGENET": np.array([0.229, 0.224, 0.225]),
    "CALTECH101": np.array([1.0, 1.0, 1.0]),
    "CALTECH256": np.array([1.0, 1.0, 1.0]),
    "CIFAR10_0.1": np.array([0.2023, 0.1994, 0.2010]),
    "CIFAR10_0.25": np.array([0.2023, 0.1994, 0.2010]),
    "CIFAR10_0.5": np.array([0.2023, 0.1994, 0.2010]),
    "CIFAR10": np.array([0.2023, 0.1994, 0.2010]),
    "CIFAR100": np.array([0.2675, 0.2565, 0.2761]),
    "CHESTXRAY14": np.array([0.229, 0.224, 0.225]),
    "SUN397": np.array([1.0, 1.0, 1.0]),
    "AIRCRAFT": np.array([1.0, 1.0, 1.0]),
    "BIRDSNAP": np.array([1.0, 1.0, 1.0]),
    "FLOWERS": np.array([1.0, 1.0, 1.0]),
    "FOOD": np.array([1.0, 1.0, 1.0]),
    "PETS": np.array([1.0, 1.0, 1.0]),
    "STANFORD_CARS": np.array([1.0, 1.0, 1.0])
}

DS_TO_NCLASSES = {
    "IMAGENET": 1000,
    "CIFAR10_0.1": 10,
    "CIFAR10_0.25": 10,
    "CIFAR10_0.5": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "CHESTXRAY14": 2,
    "SUN397": 397,
    "AIRCRAFT": 100,
    "BIRDSNAP": 500,
    "FLOWERS": 102,
    "FOOD": 101,
    "PETS": 37,
    "STANFORD_CARS": 196,
    "CALTECH101": 101,
    "CALTECH256": 257
}

class DataModule():
    def __init__(self) -> None:
        self.setup()

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

class CIFAR10DataModule(DataModule):

    def __init__(self, data_dir: str, batch_size: int = 128, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers=num_workers
        self.normalization = cifar10_normalization()
        self.train_transform = transforms.Compose([TRAIN_TRANSFORMS(32), self.normalization])
        self.test_transform = transforms.Compose([TEST_TRANSFORMS(32), self.normalization])
        super().__init__()


    def setup(self, stage=None):
        self.cifar_train = torchvision.datasets.CIFAR10(self.data_dir, download=False, train=True,
                                                        transform=self.train_transform)
        self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, download=False, train=False,
                                                        transform=self.test_transform)
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class TransferFFCVDataModule(DataModule):

    def __init__(self, ds_name: str,
                    train_path: str,
                    val_path: str,
                    batch_size: int = 128,
                    num_workers=4,
                    quasi_random=False,
                    custom_img_transform=[],
                    custom_label_transform=[],
                    resolution=224,
                    decoder_train='simple',
                    decoder_val='simple'
                    ):
        DATASET_MEAN = DS_TO_MEAN[ds_name.upper()]
        DATASET_STD = DS_TO_STD[ds_name.upper()]

        self.decoder_train = decoder_train
        self.decoder_val = decoder_val
        self.resolution = resolution

        self.loader_args = {
            'ds_name': ds_name,
            'train_path': train_path,
            'val_path': val_path,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'quasi_random': quasi_random,
            'dataset_mean': DATASET_MEAN,
            'dataset_std': DATASET_STD,
            'custom_img_transform': custom_img_transform,
            'custom_label_transform': custom_label_transform,
        }
        super().__init__()

    def setup(self, stage=None):
        self.train_loader, self.val_loader = \
            ffcv_utils.get_ffcv_loader('train',
                img_decoder=IMAGE_DECODERS[self.decoder_train](self.resolution), **self.loader_args), \
            ffcv_utils.get_ffcv_loader('val',
                img_decoder=IMAGE_DECODERS[self.decoder_val](self.resolution), **self.loader_args)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return self.val_loader
