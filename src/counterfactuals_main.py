# from wandb import Config
import copy
import os
import uuid

import git
import torch
import torch.nn as nn
import torchvision.models as torch_models
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import numpy as np
import class_influence_utils
import config_parse_utils
import datasets
from eval_utils import evaluate_model
from ffcv_aug import SelectLabel
from models import TransferNetwork
from trainer import LightWeightTrainer
from transfer_sections import add_transfer_args

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


Section("training", "training arguments").params(
    num_workers=Param(int, "number of workers", default=8),
    batch_size=Param(int, "batch size", default=512),
    exp_name=Param(str, "experiment name", default=""),
    transfer_task=Param(And(str, OneOf(["ALL", "FIXED", "FULL"])), "transfer task", default="ALL"),
    epochs=Param(int, "max epochs", default=60),
    lr=Param(float, "learning rate", default=0.1),
    weight_decay=Param(float, "weight decay", default=1e-4),
    momentum=Param(float, "SGD momentum", default=0.9),
    lr_scheduler=Param(And(str, OneOf(["steplr", "multisteplr", "cyclic"])), "learning rate scheduler", default="steplr"),
    step_size=Param(int, "step size", default=30),
    lr_milestones=Param(str, "learning rate milestones (comma-separated list of learning rate milestones, e.g. 10,20,30)", default=""),
    lr_peak_epoch=Param(int, "lr_peak_epoch for cyclic lr schedular", default=5),
    gamma=Param(float, "SGD gamma", default=0.1),
    label_smoothing=Param(float, "label smoothing", default=0.0),
    disable_logging=Param(bool, "disable logging", default=False),
    supercloud=Param(bool, "use supercloud", default=False),
    data_root=Param(str, "data root dir", default="/mnt/nfs/datasets/transfer_datasets"),
    decoder=Param(str, "FFCV image decoder.", default='random_resized_crop'),
    granularity=Param(And(str, OneOf(["global", "per_class"])), "Accuracy: global vs per class.", default='global'),
    eval_epochs=Param(int, "Evaluate every n epochs.", default=5),
    # specific CF args    
)

Section("counterfactual", "counterfactual args").params(
    cf_type=Param(str, OneOf(['CLASS', 'EXAMPLE']), default='CLASS'),
    cf_target_dataset=Param(str, "Set in order to override the transfer dataset used", required=True),
    cf_infl_order_file=Param(str, "Numpy file of class/example influences from most negative to most positive", required=True),
    cf_num_classes_min=Param(int, "Minimum number of classes to EXCLUDE", default=0),
    cf_num_classes_max=Param(int, "Maximum number of classes to EXCLUDE", default=950),
    cf_num_classes_step=Param(int, "Step size for classes to EXCLUDE", default=50),
    cf_num_examples_min=Param(int, "Minimum number of examples to EXCLUDE", default=0),
    cf_num_examples_max=Param(int, "Maximum number of examples to EXCLUDE", default=1281000),
    cf_num_examples_step=Param(int, "Step size for examples to EXCLUDE", default=50000),
    cf_order=Param(And(str, OneOf(["TOP", "BOTTOM", "ALL"])), "top first, bottom first, or both directions", default="ALL"),
)

Section("model", "model architecture arguments").params(
    arch=Param(str, "architecture to train", default="resnet18"),
    pretrained=Param(bool, "Pytorch Pretrained", default=False),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0),
    val_res=Param(int, 'validation resolution', default=224),
)

Section("data", "data arguments").params(
    dataset=Param(str, "source dataset", default="imagenet"),
    train_path=Param(str, "path of training loader", default="/home/gridsan/groups/robustness/datasets/data-transfer/imagenet-train-256.pxy"),
    val_path=Param(str, "path of validation loader", default="/home/gridsan/groups/robustness/datasets/data-transfer/imagenet-val-256.pxy"),
    num_classes=Param(int, "number of classes from source dataset", default=500),
    num_examples=Param(int, "number of examples from source dataset", default=-1),
)

Section("out", "output arguments").params(
    output_pkl_dir=Param(str, "output pickle file", default="output_pkl_new")
)

add_transfer_args()

def train_source_model(args, logdir, exclude_list, cf_type):
    root_dir = args.data_root
    train_path = os.path.join(root_dir, args.train_path)
    val_path = os.path.join(root_dir, args.val_path)

    training_args = {'epochs': args.epochs, 'lr': args.lr,
                    'weight_decay': args.weight_decay, 'momentum': args.momentum,
                    'lr_scheduler': args.lr_scheduler, 'step_size': args.step_size,
                    'lr_milestones': args.lr_milestones, 'gamma': args.gamma,
                    'label_smoothing': args.label_smoothing,'lr_peak_epoch': args.lr_peak_epoch,
                    'eval_epochs': args.eval_epochs}

    res_args = {'min_res': args.min_res, 'max_res': args.max_res, 
                'start_ramp': args.start_ramp, 'end_ramp': args.end_ramp,}

    trainer = LightWeightTrainer(training_args, "imagenet", logdir, res_args=res_args, 
                                 enable_logging=not args.disable_logging, granularity=args.granularity)

    model = torch_models.__dict__[args.arch](num_classes=1000, pretrained=args.pretrained).cuda()
    
    common_args = {
        'train_path': train_path,
        'val_path': val_path, 'batch_size': args.batch_size,
        'num_workers': args.num_workers, 'quasi_random': args.supercloud,
        'exclude_file': exclude_list, 'resolution': args.val_res, 'decoder': args.decoder
    }

    classes_to_keep, examples_to_keep = None, None
    if cf_type == 'CLASS':
        train_loader, val_loader, classes_to_keep = class_influence_utils.get_subset_class_workers(
            num_classes=args.num_classes, **common_args)
    elif cf_type == 'EXAMPLE':
        train_loader, val_loader, examples_to_keep = class_influence_utils.get_subset_example_workers(
            num_examples=args.num_examples, **common_args)
        
    trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader)
    source_results = evaluate_model(model, val_loader, num_classes=1000, granularity=args.granularity)

    return model, source_results, classes_to_keep, examples_to_keep

def apply_transfer(name, logdir, transfer_args, transfer_model, datamodule, num_classes, granularity):

    trainer = LightWeightTrainer(
        transfer_args, name, logdir, 
        enable_logging=False, granularity=granularity)
    trainer.fit(transfer_model, datamodule.train_dataloader(), datamodule.val_dataloader())
    return evaluate_model(transfer_model, datamodule.val_dataloader(), 
        num_classes=num_classes, granularity=granularity, name=name)


if __name__ == "__main__":
    in_args, transfer_configs = config_parse_utils.process_args_and_config()
    in_args = copy.deepcopy(in_args)
    transfer_config_path = transfer_configs[DS_TO_CONFIG[in_args.cf_target_dataset]]
    data_root = in_args.data_root            
    transfer_args = config_parse_utils.update_config_with_transfer_config(transfer_config_path)
                
    os.makedirs(in_args.output_pkl_dir, exist_ok=True)
    assert not in_args.pretrained

    EXP_NAME = str(uuid.uuid4()) if not in_args.exp_name else in_args.exp_name
    if in_args.cf_order == 'ALL':
        cf_orders = ['TOP', 'BOTTOM']
    else:
        cf_orders = [in_args.cf_order]
        
    repo = git.Repo(search_parent_directories=True)
    args_dict = {'in_args': vars(in_args), 'transfer_args': vars(transfer_args)}
    args_dict['commit'] = repo.head.object.hexsha
            
    infl_orders = np.load(in_args.cf_infl_order_file)
    if in_args.cf_type == 'CLASS':
        trajectory = np.arange(in_args.cf_num_classes_min, in_args.cf_num_classes_max+1, in_args.cf_num_classes_step)
    elif in_args.cf_type == 'EXAMPLE':
        trajectory = np.arange(in_args.cf_num_examples_min, in_args.cf_num_examples_max+1, in_args.cf_num_examples_step)
    else:
        raise NotImplementedError
    
    all_out = {}
    
    for cf_order in cf_orders:
        order_out = {}
        for K in trajectory:
            print(f"---- {cf_order} ------ {K} ---------", flush=True)
            if cf_order == 'BOTTOM':
                exclude_list = infl_orders[:K]
            elif cf_order == 'TOP':
                exclude_list = infl_orders[::-1][:K]
            else:
                raise NotImplementedError
                    
                
            # Source Model
            model, source_results, classes_to_keep, examples_to_keep = train_source_model(in_args, logdir=EXP_NAME,
                                                                        exclude_list=exclude_list, cf_type=in_args.cf_type)
            # Extract backbone
            in_dim = model.fc.in_features
            backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
            out = {'classes_to_keep': classes_to_keep, 'source_results': source_results,
                   'transfer_ds_name': transfer_args.transfer_dataset, 'args': args_dict, 'K': K,}

            logdir = "_".join([EXP_NAME, transfer_args.transfer_dataset])


            # -------------- Transfer Dataset --------------------
            if transfer_args.transfer_ffcv:
                custom_label_transform = [SelectLabel(7)] if transfer_args.transfer_dataset.upper()=="CHESTXRAY14" else []
                transfer_ds = datasets.TransferFFCVDataModule(ds_name=transfer_args.transfer_dataset,
                                                        train_path=os.path.join(data_root, transfer_args.transfer_path_train),
                                                        val_path=os.path.join(data_root, transfer_args.transfer_path_val),
                                                        batch_size=transfer_args.batch_size,
                                                        num_workers=transfer_args.num_workers,
                                                        quasi_random=transfer_args.supercloud,
                                                        resolution=transfer_args.val_res,
                                                        decoder_train=transfer_args.decoder_train,
                                                        decoder_val=transfer_args.decoder_val,
                                                        custom_label_transform=custom_label_transform)

            else:
                if transfer_args.transfer_dataset == "CIFAR10":
                    transfer_ds = datasets.CIFAR10DataModule(data_dir='/mnt/nfs/datasets/cifar10',
                                                        num_workers=transfer_args.num_workers,
                                                        batch_size=transfer_args.batch_size)
                else:
                    raise NotImplementedError
                    
            t_args = {
                'epochs': transfer_args.transfer_epochs, 'lr': transfer_args.transfer_lr,
                'weight_decay': transfer_args.transfer_weight_decay, 'momentum': transfer_args.transfer_momentum,
                'lr_scheduler': transfer_args.transfer_lr_scheduler, 'step_size': transfer_args.transfer_step_size,
                'lr_milestones': transfer_args.transfer_lr_milestones, 'gamma': transfer_args.transfer_gamma,
                'label_smoothing': transfer_args.transfer_label_smoothing,
                'lr_peak_epoch': transfer_args.transfer_lr_peak_epoch,
                'eval_epochs': transfer_args.transfer_eval_epochs
            }
            if transfer_args.upsample:
                preprocess_transfer = nn.Upsample((transfer_args.val_res, transfer_args.val_res))
            else:
                preprocess_transfer = None

            if transfer_args.transfer_task == ['ALL']:
                transfer_tasks = ['FIXED', 'FULL']
            else:
                transfer_tasks = [transfer_args.transfer_task]
            num_classes = datasets.DS_TO_NCLASSES[transfer_args.transfer_dataset.upper()]
                        
            
            for transfer_task in transfer_tasks:
                if transfer_task == 'FIXED':
                    freeze_backbone = True
                else:
                    freeze_backbone = False
                print(F'==>[Transfer to {transfer_args.transfer_dataset}, {transfer_task} finetuning]')
                transfer_model = TransferNetwork(num_classes=num_classes, backbone_out_dim=in_dim,
                    preprocess=preprocess_transfer, freeze_backbone_bn=False, backbone=copy.deepcopy(backbone), 
                    freeze_backbone=freeze_backbone).cuda()
                transfer_results = apply_transfer(transfer_args.transfer_dataset, logdir, t_args,
                                                  transfer_model, transfer_ds, num_classes=num_classes,
                                                  granularity=transfer_args.transfer_granularity)
                if transfer_task == 'FIXED':
                    out['transfer_results'] = transfer_results
                else:
                    out['transfer_results_full'] = transfer_results
            order_out[K] = out
        all_out[cf_order] = order_out

    output_pkl_file = os.path.join(in_args.output_pkl_dir, EXP_NAME + '.pt')
    torch.save(all_out, output_pkl_file)

    print("==>[Job successfully done.]")
