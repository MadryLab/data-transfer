import os 
import numpy as np
from label_maps import CLASS_DICT
import torch
from collections import defaultdict
from tqdm import tqdm
import random
from sklearn.linear_model import Ridge, Lasso

def recursive_compress(arr):
    result = {k: [u[k] for u in arr] for k in arr[0]}
    for k in result.keys():
        if torch.is_tensor(result[k][0]):
            result[k] = torch.cat(result[k])
        elif isinstance(result[k][0], dict):
            result[k] = recursive_compress(result[k])
        else:
            result[k] = torch.tensor(result[k])
    return result

# READING FROM MMAPPED FILES ====================
SCHEMA_ENTRIES = {
    'softmax_logits': 'float16',
    'raw_logits': 'float16',
    'softmax_margins': 'float16',
    'raw_margins': 'float16',
    'is_corrects': 'bool_',
    'predictions': 'int64',
    'acc': 'float16'
}

def memmap_read(k, logdir, index=None):
    this_filename = os.path.join(logdir, f"{k}.npy")
    # make sure that the mmap actually exists
    assert os.path.exists(this_filename)

    mmap = np.lib.format.open_memmap(this_filename, mode='r')
    if index is not None:
        return mmap[index]
    else:
        return mmap
    
def get_mm_flags(memmaps_dir):
    return memmap_read("_completed", memmaps_dir, None)

def get_mm_masks(mask_file):
    return np.lib.format.open_memmap(mask_file, mode='r')

def get_mm_values(memmaps_dir, keyword):
    assert keyword in SCHEMA_ENTRIES
    result = memmap_read(keyword, memmaps_dir, None)
    assert result.dtype == SCHEMA_ENTRIES[keyword]
    return result

class MMFileDataset():
    def __init__(self, keyword, memmaps_dir, mask_file="/mnt/nfs/home/siraj/extracted/train_masks.npy"):
        self.masks = get_mm_masks(mask_file)
        self.flags = get_mm_flags(memmaps_dir)
        self.num_models = self.flags.shape[0]
        self.active_indices = np.arange(self.num_models)[self.flags]
        self.logits = get_mm_values(memmaps_dir, keyword)
        
    def __len__(self):
        return len(self.active_indices)
    
    def __getitem__(self, idx):
        active_idx = self.active_indices[idx]
        mask = torch.tensor(self.masks[active_idx])
        logits = torch.tensor(self.logits[active_idx])
        return mask, logits

# READING FROM SUMMARY FILES ====================
def read_summary_file(name, ds):
    return torch.load(os.path.join(name, f"{ds}.pt"))

class SummaryFileDataSet():
    def __init__(self, summary_folders, dataset, influence_key, keyword):
        if isinstance(summary_folders, str):
            summary_folders = [summary_folders]
            
        data = recursive_compress([read_summary_file(d, dataset) for d in summary_folders])
        self.data = data
        self.class_binary_masks = data['class_binary_mask']
        self.influence_key = influence_key
        self.keyword = keyword
    
    def __len__(self):
        return self.class_binary_masks.shape[0]
    
    def __getitem__(self, idx):
        class_mask = self.class_binary_masks[idx]
        logits = self.data[self.keyword][self.influence_key][idx]
        return class_mask, logits
    
# READING FROM SUMMARY FILES ====================

class BatchInfluencesMeter():
    def __init__(self, target_N, source_N, div=False):
        self.results = {
            'pos_infl': torch.zeros((target_N, source_N)).float(),
            'neg_infl': torch.zeros((target_N, source_N)).float(),
            'pos_denom': torch.zeros((1, source_N)).float(),
            'neg_denom': torch.zeros((1, source_N)).float(),
        }
        self.source_N = source_N
        self.target_N = target_N
        self.div = div
    def update(self, logits, binary_masks):
        # logits:  Num_Models x Target_N
        # binary_masks: Num_Models x Source_N
        logits = logits.T.unsqueeze(-1).float() # Target_N x Num_Models x 1
        class_mask = binary_masks.float().cuda()
        neg_class_mask = 1 - class_mask
        
        self.results['pos_denom'] += class_mask.sum(axis=0).unsqueeze(0).cpu()
        self.results['neg_denom'] += neg_class_mask.sum(axis=0).unsqueeze(0).cpu()
        
        class_mask = class_mask.T
        neg_class_mask = neg_class_mask.T
        
        if self.div:
            step_size = 1000
            for i in tqdm(range(0, self.target_N, step_size)):
                logit_faction = logits[i:i+step_size]
                self.results['pos_infl'][i:i+step_size] += (class_mask @ logit_faction).squeeze(-1).cpu() 
                self.results['neg_infl'][i:i+step_size] += (neg_class_mask @ logit_faction).squeeze(-1).cpu() 
        else:
            logits = logits.cuda()
            self.results['pos_infl'] += (class_mask @ logits).squeeze(-1).cpu() # Target_N x Source_N
            self.results['neg_infl'] += (neg_class_mask @ logits).squeeze(-1).cpu() # Target_Classes x Source_Classes

        
    def calculate(self):
        pos_infl =  self.results['pos_infl']/self.results['pos_denom']
        neg_infl = self.results['neg_infl']/self.results['neg_denom']
        return (pos_infl - neg_infl).cpu().numpy()
    
    
def batch_calculate_influence(dl, target_N, source_N, div=False):
    meter = BatchInfluencesMeter(target_N, source_N, div=div)
    for batch in tqdm(dl):
        binary_mask, logits = batch
        binary_mask = binary_mask.cuda()
        logits = logits.cuda()
        meter.update(logits=logits, binary_masks=binary_mask)
    return meter.calculate()

