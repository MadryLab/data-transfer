import numpy as np
import torch as ch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
# from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, RandomHorizontalFlip, Squeeze,
                             ToDevice, ToTensor)
# import antialiased_cnns
from ffcv.transforms.ops import ToTorchImage
from torchvision.transforms import Normalize


def get_ffcv_loader(split='train',
                    batch_size=1024,
                    num_workers=80,
                    ds_name="imagenet",
                    train_path='',
                    val_path='',
                    indices=None,
                    img_decoder=SimpleRGBImageDecoder(),
                    custom_img_transform=[], # pre normalization, on cpu
                    custom_label_transform=[],
                    shuffle=None,
                    drop_last=None,
                    quasi_random=False,
                    dataset_mean=np.array([0.0, 0.0, 0.0]),
                    dataset_std=np.array([1.0, 1.0, 1.0]),
                    pipeline_keys=['image', 'label']
                ):

    RANDOM_ORDER = OrderOption.QUASI_RANDOM if quasi_random else OrderOption.RANDOM

    if shuffle is not None:
        order = RANDOM_ORDER if shuffle == True else OrderOption.SEQUENTIAL
    else:
        order = RANDOM_ORDER if split == 'train' else OrderOption.SEQUENTIAL

    if drop_last is None:
        drop_last = True if split == 'train' else False

    image_pipeline = [img_decoder,
                      *custom_img_transform,
                      ToTensor(),
                      ToDevice(ch.device('cuda'), non_blocking=True),
                      ToTorchImage(),
                      Convert(ch.float16),
                      Normalize((dataset_mean * 255).tolist(), (dataset_std * 255).tolist())]

    if split == "train":
        image_pipeline.insert(1, RandomHorizontalFlip())

    if ds_name.upper() == "CHESTXRAY14":
        label_pipeline= [NDArrayDecoder(),
                        *custom_label_transform,
                        ToTensor(),
                        Squeeze(),
                        Convert(ch.int64),
                        ToDevice(ch.device('cuda'), non_blocking=True)]
    else:
        label_pipeline= [IntDecoder(),
                        ToTensor(),
                        Squeeze(),
                        *custom_label_transform,
                        ToDevice(ch.device('cuda'), non_blocking=True)]

    print(split, order, drop_last)
    pipelines = {'image': image_pipeline,'label': label_pipeline}
    pipelines = {k: v if k in pipeline_keys else None for k,v in pipelines.items()}
    return Loader(fname=train_path if split == 'train' else val_path,
                  batch_size=batch_size,
                  num_workers=num_workers,
                  order=order,
                  os_cache=not quasi_random,
                  indices=indices,
                  pipelines=pipelines,
                  drop_last=drop_last)
