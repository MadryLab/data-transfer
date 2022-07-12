import yaml
import sys
sys.path.append('../src')

import ffcv_utils
import os
import numpy as np
from decoders_and_transforms import IMAGE_DECODERS
import tqdm
import torch
from ffcv_aug import SelectLabel

DS_TO_CONFIG = {
    'CIFAR100': 'cifar100', 
    'CARS': 'stanford_cars',
    'FLOWERS': 'flowers', 
    'CALTECH256': 'caltech256', 
    'BIRDSNAP': 'birdsnap', 
    'FOOD': 'food', 
    'CIFAR10': 'cifar10',
    'SUN397': 'SUN397', 
    'CIFAR10_0.25': 'cifar10_0_25', 
    'AIRCRAFT': 'aircraft', 
    'CIFAR10_0.1': 'cifar10_0_1', 
    'CALTECH101': 'caltech101', 
    'CIFAR10_0.5': 'cifar10_0_5', 
    'PETS': 'pets',
    'CHESTXRAY14': 'chestxray14',
    'IMAGENET': 'imagenet',
}

def read_yaml(f):
    with open(f, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
        
def get_transfer_config(base_config_name, ds_name):
    base_config = read_yaml(base_config_name)
    transfer_config_yamls = base_config['transfer_configs']
    transfer_configs = {}
    for k, v in transfer_config_yamls.items():
        transfer_configs[k] = read_yaml(os.path.join('..', v))
    print(transfer_configs.keys())
    t_config = transfer_configs[DS_TO_CONFIG[ds_name]]
    return t_config, base_config

def get_image_loader(base_config_name, ds_to_analyze, indices=None, pipeline_keys=['image', 'label'], split='val', batchsize=100):
    t_config, base_config = get_transfer_config(base_config_name, ds_to_analyze)
    data_root = base_config['training']['data_root']
    ds_name = t_config['transfer_data']['transfer_dataset']
    loader_args = {
        'ds_name': ds_name,
        'train_path': os.path.join(data_root, t_config['transfer_data']['transfer_path_train']),
        'val_path': os.path.join(data_root, t_config['transfer_data']['transfer_path_val']),
        'batch_size': batchsize,
        'num_workers': 1,
        'quasi_random': False,
        'dataset_mean': np.array([0, 0, 0]),
        'dataset_std': np.array([1, 1, 1]),
        'img_decoder': IMAGE_DECODERS['center_crop_256'](224),
        'indices': indices,
        'shuffle': False,
        'pipeline_keys': pipeline_keys,
        'custom_label_transform': [SelectLabel(7)] if ds_name.upper()=="CHESTXRAY14" else [],
        'drop_last': False,
    }
    loader = ffcv_utils.get_ffcv_loader(split, **loader_args)
    return loader

def get_train_image_loader(base_config_name, ds_to_analyze, indices=None, pipeline_keys=['image', 'label'], batchsize=100):
    return get_image_loader(base_config_name, ds_to_analyze, indices, pipeline_keys, split='train', batchsize=batchsize)

def get_train_labels(base_config_name, ds_to_analyze):
    loader = get_train_image_loader(base_config_name, ds_to_analyze, pipeline_keys=['label'])
    outputs = []
    for batch in tqdm.tqdm(loader):
        outputs.append(batch[0].cpu())
    return torch.cat(outputs)

def get_val_image_loader(base_config_name, ds_to_analyze, indices=None, pipeline_keys=['image', 'label'], batchsize=100):
    return get_image_loader(base_config_name, ds_to_analyze, indices, pipeline_keys, split='val', batchsize=batchsize)

def get_val_labels(base_config_name, ds_to_analyze):
    loader = get_val_image_loader(base_config_name, ds_to_analyze, pipeline_keys=['label'])
    outputs = []
    for batch in tqdm.tqdm(loader):
        outputs.append(batch[0].cpu())
    return torch.cat(outputs)