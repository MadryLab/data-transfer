# from wandb import Config
import copy
import os
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from tqdm import tqdm

from threading import Lock
lock = Lock()

def get_margin(out, ys):
    logit_ = out[torch.arange(out.shape[0]), ys].clone()
    out[torch.arange(out.shape[0]), ys] = -np.inf
    new_max = out.max(-1).values
    margin = logit_ - new_max
    return logit_, margin

def get_auroc(logits, target):
    logits_np = torch.softmax(logits, dim=1).cpu().detach().numpy()[:, 1]
    target_np = target.cpu().detach().numpy()
    score = roc_auc_score(target_np, logits_np)
    return 100 * score

def evaluate_model(model, loader, num_classes, granularity="global", name=None):

    assert granularity in ["global", "per_class"]
    if granularity == "global":
        per_class = False
    else:
        per_class = True

    is_train = model.training
    model.eval().cuda()

    if per_class:
        cm = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        softmax = nn.Softmax(dim=-1)
        softmax_logits, raw_logits = [], []
        softmax_margins, raw_margins = [], []
        is_corrects = []
        predictions = []
        all_targets = []
        all_raw_out = []
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            with autocast():
                with lock:
                    raw_out = model(x)
                softmax_out = softmax(raw_out)

                if name == 'CHESTXRAY14':
                    all_targets.append(y)
                    all_raw_out.append(copy.deepcopy(raw_out))

                if per_class:
                    preds = softmax_out.max(1)[1]
                    if cm.device != y.device:
                        cm = cm.to(y.device)
                    for t, p in zip(y.view(-1), preds.view(-1)):
                        cm[t.long(), p.long()] += 1

                max_class = softmax_out.argmax(-1)
                predictions.append(max_class)
                is_corrects.append((max_class == y).cpu())

                logit, margin = get_margin(raw_out, y)
                raw_logits.append(logit.cpu())
                raw_margins.append(margin.cpu())
                softmax_logit, softmax_margin = get_margin(softmax_out, y)
                softmax_logits.append(softmax_logit.cpu())
                softmax_margins.append(softmax_margin.cpu())


        if per_class:
            class_acc = cm.diag()/cm.sum(1)
            class_acc = 100*torch.nan_to_num(class_acc, nan=0.0).cpu().detach()

        result = {
            'softmax_logits': torch.cat(softmax_logits).half(),
            'raw_logits': torch.cat(raw_logits).half(),
            'softmax_margins': torch.cat(softmax_margins).half(),
            'raw_margins': torch.cat(raw_margins).half(),
            'is_corrects': torch.cat(is_corrects),
            'predictions': torch.cat(predictions),
        }

        result['acc'] = result['is_corrects'].half().mean() * 100
        print("Accuracy: ", result['acc'].item())

        if name == 'CHESTXRAY14':
            all_raw_out = torch.cat(all_raw_out)
            all_targets = torch.cat(all_targets)
            auroc = get_auroc(all_raw_out, all_targets)
            result.update({'auroc': auroc})
            print("AUROC: ", auroc)

        if per_class:
            result.update({'class_acc': class_acc})
            print("MeanPerClass Accuracy: ", class_acc.mean().item())

    model.train(is_train)
    return result
