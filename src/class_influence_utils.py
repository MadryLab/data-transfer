import pickle as pkl

import numpy as np
import torch

import ffcv_utils
from decoders_and_transforms import IMAGE_DECODERS
import tqdm

def get_imagenet_labels(train_path, val_path):
    loader_args = {
        'ds_name': 'imagenet', 'train_path': train_path, 'val_path': val_path,
        'batch_size': 100, 'num_workers': 1, 'quasi_random': False,
        'dataset_mean': np.array([0, 0, 0]), 'dataset_std': np.array([1, 1, 1]),
        'img_decoder': IMAGE_DECODERS['center_crop_256'](224),
        'indices': None, 'shuffle': False, 'pipeline_keys': ['label'], 'drop_last': False,
    }
    label_loaders = {
        'train': ffcv_utils.get_ffcv_loader('train', **loader_args),
        'val': ffcv_utils.get_ffcv_loader('val', **loader_args),
    }
    label_results = {}
    for k, loader in label_loaders.items():
        outputs = []
        for batch in tqdm.tqdm(loader):
            outputs.append(batch[0].cpu())
        outputs = torch.cat(outputs)
        label_results[k] = outputs
    print({k: len(label_results[k]) for k in label_results.keys()})
    return label_results

def get_indices_subset(subset_pkl, num_classes, exclude_file=None):
    out = subset_pkl

    if exclude_file is not None:
        if isinstance(exclude_file, str):
            print("getting excluded classes from", exclude_file)
            classes_to_exclude = np.load(exclude_file)
        else:
            assert isinstance(exclude_file, np.ndarray)
            classes_to_exclude = exclude_file
        print(f'Excluding {classes_to_exclude.shape[0]} classes')
        classes_to_keep = torch.tensor([u for u in np.arange(1000) if u not in classes_to_exclude]).long()
        if num_classes == -1:
            num_classes = len(classes_to_keep)
        else:
            assert len(classes_to_keep) == num_classes
    else:
        classes_to_keep = torch.randperm(1000)[:num_classes].numpy()
        classes_to_keep.sort()

    train_subset = np.array(out['train'])
    train_indices = np.arange(len(train_subset))[np.in1d(train_subset, classes_to_keep)]

    val_subset = np.array(out['val'])
    val_indices = np.arange(len(val_subset))[np.in1d(val_subset, classes_to_keep)]

    idx_map = torch.ones(1000)*-1
    for i, c in enumerate(classes_to_keep):
        idx_map[c] = i
    return classes_to_keep, idx_map, train_indices, val_indices

def get_examples_subset(subset_pkl, num_examples, exclude_file=None):
    out = subset_pkl
    train_subset = np.array(out['train'])
    N_train = len(train_subset)

    if exclude_file is not None:
        if isinstance(exclude_file, str):
            print("Getting excluded examples from", exclude_file)
            examples_to_exclude = np.load(exclude_file)
        else:
            assert isinstance(exclude_file, np.ndarray)
            examples_to_exclude = exclude_file
        examples_to_keep = np.arange(N_train)[~np.in1d(np.arange(N_train), examples_to_exclude)]
        if num_examples == -1:
            num_examples = len(examples_to_keep)
        else:
            assert len(examples_to_keep) == num_examples
    else:
        examples_to_keep = torch.randperm(N_train)[:num_examples].numpy()
        examples_to_keep.sort()

    
    train_indices = examples_to_keep

    val_subset = np.array(out['val'])
    val_indices = np.arange(len(val_subset))
    return examples_to_keep, train_indices, val_indices
    


class SubsetLabelTransform(torch.nn.Module):
    def __init__(self, idx_map):
        super().__init__()
        self.idx_map = idx_map
        
    def forward(self, y):
        return self.idx_map[y]
    
def _get_loaders(batch_size, num_workers, train_path, val_path, shuffle, 
                 drop_last, quasi_random, resolution, decoder, train_indices, val_indices):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    common_args = {
        'ds_name': "imagenet",
        'batch_size': batch_size,
        'num_workers': num_workers,
        'train_path': train_path,
        'val_path': val_path,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'quasi_random': quasi_random,
        'dataset_mean': IMAGENET_MEAN,
        'dataset_std': IMAGENET_STD,
        # 'custom_label_transform': [SubsetLabelTransform(idx_map)],
    }

    train_loader = ffcv_utils.get_ffcv_loader(split='train', 
                    img_decoder=IMAGE_DECODERS[decoder](resolution), 
                    indices=train_indices, **common_args)
    val_loader = ffcv_utils.get_ffcv_loader(split='val', 
                    img_decoder=IMAGE_DECODERS['center_crop_256'](resolution), 
                    indices=val_indices, **common_args)
    return train_loader, val_loader


def get_subset_class_workers(num_classes, train_path, val_path, batch_size, num_workers,
                            shuffle=None, drop_last=None, quasi_random=False,
                            exclude_file=None, resolution=224, decoder='random_resized_crop'):
    # saves the class mask to out_dir/class_mask.npy
    subset_pkl = get_imagenet_labels(train_path=train_path, val_path=val_path)
    classes_to_keep, idx_map, train_indices, val_indices = get_indices_subset(subset_pkl, num_classes, exclude_file)

    train_loader, val_loader = _get_loaders(batch_size=batch_size, num_workers=num_workers, train_path=train_path,
                                            val_path=val_path, shuffle=shuffle, drop_last=drop_last, 
                                            quasi_random=quasi_random, resolution=resolution, decoder=decoder,
                                            train_indices=train_indices, val_indices=val_indices)
    return train_loader, val_loader, classes_to_keep

def get_subset_example_workers(num_examples, train_path, val_path, batch_size, num_workers,
                               shuffle=None, drop_last=None, quasi_random=False,
                               exclude_file=None, resolution=224, decoder='random_resized_crop'):
    # saves the class mask to out_dir/class_mask.npy
    subset_pkl = get_imagenet_labels(train_path=train_path, val_path=val_path)
    examples_to_keep, train_indices, val_indices = get_examples_subset(subset_pkl, num_examples, exclude_file)

    train_loader, val_loader = _get_loaders(batch_size=batch_size, num_workers=num_workers, train_path=train_path,
                                            val_path=val_path, shuffle=shuffle, drop_last=drop_last, 
                                            quasi_random=quasi_random, resolution=resolution, decoder=decoder,
                                            train_indices=train_indices, val_indices=val_indices)
    return train_loader, val_loader, examples_to_keep
