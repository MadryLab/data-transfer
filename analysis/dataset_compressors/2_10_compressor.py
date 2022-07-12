import torch
import os
import tqdm


ROOT = "/home/gridsan/groups/robustness/data-transfer-exps/models_2_10_22_50_percenters"
save_path = "/home/gridsan/groups/robustness/data-transfer-exps/summary_files/models_2_10_22_50_percenters_summaries"

DATASETS = ['AIRCRAFT', 'BIRDSNAP', 'CALTECH101', 'CALTECH256', 'CARS', 
 'CHESTXRAY14', 'CIFAR10', 'CIFAR100', 'CIFAR10_0.1', 
 'CIFAR10_0.25', 'CIFAR10_0.5', 'FLOWERS', 'FOOD', 'PETS', 'SUN397']

def create_dict(out):
    class_binary_mask = torch.zeros(1000).int()
    class_binary_mask[out['classes_to_keep']] = 1
    keys_to_keep = ['is_corrects', 'predictions', 'softmax_logits', 'acc']
    result_dict = {'class_binary_mask': class_binary_mask, 'transfer_results_full': {}, 'transfer_results': {}}
    for k in keys_to_keep:
        for k2 in ['transfer_results_full', 'transfer_results']:
            result_dict[k2][k] = out[k2][k]
    return result_dict

def get_ds_name(path_name):
    return '_'.join(path_name.split('_')[:-1])

def recursive_compress(arr):
    result = {k: [u[k] for u in arr] for k in arr[0]}
    for k in result.keys():
        if torch.is_tensor(result[k][0]):
            result[k] = torch.stack(result[k])
        elif isinstance(result[k][0], dict):
            result[k] = recursive_compress(result[k])
        else:
            result[k] = torch.tensor(result[k])
    return result
    
def compress_dataset(ds_name):
    dicts = []
    subset_paths = [p for p in os.listdir(ROOT) if get_ds_name(p) == ds_name]
    for path in tqdm.tqdm(subset_paths):
        out = torch.load(os.path.join(ROOT, path), map_location='cpu')
        dicts.append(create_dict(out))
    return recursive_compress(dicts)

for DS in DATASETS:
    print(DS)
    out = compress_dataset(DS)
    torch.save(out, os.path.join(save_path, f"{DS}.pt"))