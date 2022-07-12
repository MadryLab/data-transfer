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

import class_influence_utils
import config_parse_utils
import datasets
from eval_utils import evaluate_model
from ffcv_aug import SelectLabel
from models import TransferNetwork
from trainer import LightWeightTrainer
from transfer_sections import add_transfer_args

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
    disable_logging=Param(int, "disable logging", default=False),
    supercloud=Param(bool, "use supercloud", default=False),
    data_root=Param(str, "data root dir", default="/mnt/nfs/datasets/transfer_datasets"),
    decoder=Param(str, "FFCV image decoder.", default='random_resized_crop'),
    granularity=Param(And(str, OneOf(["global", "per_class"])), "Accuracy: global vs per class.", default='global'),
    eval_epochs=Param(int, "Evaluate every n epochs.", default=5),
    transfer_dataset_override=Param(str, "Set in order to override the transfer dataset used", default=''),
)

Section("model", "model architecture arguments").params(
    arch=Param(str, "architecture to train", default="resnet18"),
    pretrained=Param(int, "Pytorch Pretrained", default=False),
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
    exclude_file=Param(str, "", default=""),
)

Section("out", "output arguments").params(
    output_pkl_dir=Param(str, "output pickle file", default="output_pkl_new")
)

add_transfer_args()

def train_source_model(args, logdir):
    root_dir = args.data_root
    train_path = os.path.join(root_dir, args.train_path)
    val_path = os.path.join(root_dir, args.val_path)

    training_args = {'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'momentum': args.momentum,
                    'lr_scheduler': args.lr_scheduler,
                    'step_size': args.step_size,
                    'lr_milestones': args.lr_milestones,
                    'gamma': args.gamma,
                    'label_smoothing': args.label_smoothing,
                    'lr_peak_epoch': args.lr_peak_epoch,
                    'eval_epochs': args.eval_epochs
                    }

    res_args = {
        'min_res': args.min_res,
        'max_res': args.max_res,
        'start_ramp': args.start_ramp,
        'end_ramp': args.end_ramp,
    }

    trainer = LightWeightTrainer(training_args, "imagenet", logdir, res_args=res_args, enable_logging=not args.disable_logging, granularity=args.granularity)

    model = torch_models.__dict__[args.arch](num_classes=1000, pretrained=args.pretrained).cuda()
    train_loader, val_loader, classes_to_keep = class_influence_utils.get_subset_class_workers(
                                                        num_classes=args.num_classes, train_path=train_path,
                                                        val_path=val_path, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, quasi_random=args.supercloud,
                                                        exclude_file=args.exclude_file, resolution=args.val_res,
                                                        decoder=args.decoder)

    trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader)
    source_results = evaluate_model(model, val_loader, num_classes=1000, granularity=args.granularity)

    return model, source_results, classes_to_keep


if __name__ == "__main__":
    args, transfer_configs = config_parse_utils.process_args_and_config()
    transfer_dataset_override = args.transfer_dataset_override
    os.makedirs(args.output_pkl_dir, exist_ok=True)
    assert not args.pretrained

    EXP_NAME = str(uuid.uuid4()) if not args.exp_name else args.exp_name

    model, source_results, classes_to_keep = train_source_model(args, logdir=EXP_NAME)

    # Extract backbone
    in_dim = model.fc.in_features
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    def apply_transfer(name, logdir, transfer_args, transfer_model, datamodule, num_classes, granularity):

        trainer = LightWeightTrainer(transfer_args, name, logdir, enable_logging=not args.disable_logging, granularity=granularity)
        trainer.fit(transfer_model, datamodule.train_dataloader(), datamodule.val_dataloader())
        return evaluate_model(transfer_model, datamodule.val_dataloader(), num_classes=num_classes, granularity=granularity, name=name)

    out = {'classes_to_keep': classes_to_keep,
           'source_results': source_results,
           'transfer_datasets': {}}

    for transfer_ds_name, transfer_config_path in transfer_configs.items():
        if transfer_dataset_override and transfer_ds_name.lower() != transfer_dataset_override.lower():
            continue
        data_root = args.data_root            
        args = config_parse_utils.update_config_with_transfer_config(transfer_config_path)
        # tr_args = config_parse_utils.read_yaml(transfer_config_path)
        # args = config_parse_utils.update_tr_args(args, tr_args)

        logdir = "_".join([EXP_NAME, args.transfer_dataset])

        transfer_args = {'epochs': args.transfer_epochs,
                        'lr': args.transfer_lr,
                        'weight_decay': args.transfer_weight_decay,
                        'momentum': args.transfer_momentum,
                        'lr_scheduler': args.transfer_lr_scheduler,
                        'step_size': args.transfer_step_size,
                        'lr_milestones': args.transfer_lr_milestones,
                        'gamma': args.transfer_gamma,
                        'label_smoothing': args.transfer_label_smoothing,
                        'lr_peak_epoch': args.transfer_lr_peak_epoch,
                        'eval_epochs': args.transfer_eval_epochs
                        }

        # -------------- Transfer Dataset --------------------
        if args.transfer_ffcv:
            transfer_ds = datasets.TransferFFCVDataModule(ds_name=args.transfer_dataset,
                                                    train_path=os.path.join(data_root, args.transfer_path_train),
                                                    val_path=os.path.join(data_root, args.transfer_path_val),
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    quasi_random=args.supercloud,
                                                    resolution=args.val_res,
                                                    decoder_train=args.decoder_train,
                                                    decoder_val=args.decoder_val,
                                                    custom_label_transform=[SelectLabel(7)] if args.transfer_dataset.upper()=="CHESTXRAY14" else [])

        else:
            if args.transfer_dataset == "CIFAR10":
                transfer_ds = datasets.CIFAR10DataModule(data_dir='/mnt/nfs/datasets/cifar10',
                                                    num_workers=args.num_workers,
                                                    batch_size=args.batch_size)
            else:
                raise NotImplementedError

        # Transfer Dataset frozen backbone
        if args.transfer_task in ['ALL', 'FIXED']:
            print(F'==>[Transfer to {args.transfer_dataset}]')
            preprocess_transfer = nn.Upsample((args.val_res, args.val_res)) if args.upsample else None
            transfer_model = TransferNetwork(num_classes=datasets.DS_TO_NCLASSES[args.transfer_dataset.upper()],
                                        backbone_out_dim=in_dim,
                                        backbone=copy.deepcopy(backbone),
                                        preprocess=preprocess_transfer,
                                        freeze_backbone=True,
                                        freeze_backbone_bn=False).cuda()
            transfer_results = apply_transfer(args.transfer_dataset,
                                            logdir,
                                            transfer_args,
                                            transfer_model,
                                            transfer_ds,
                                            num_classes=datasets.DS_TO_NCLASSES[args.transfer_dataset.upper()],
                                            granularity=args.transfer_granularity)
        else:
            transfer_results = {}

        # Transfer Dataset full training
        if args.transfer_task in ['ALL', 'FULL'] and args.transfer_dataset != 'IMAGENET':
            print(F'==>[Transfer to {args.transfer_dataset} full finetuning]')
            preprocess_transfer = nn.Upsample((args.val_res, args.val_res)) if args.upsample else None
            transfer_model = TransferNetwork(num_classes=datasets.DS_TO_NCLASSES[args.transfer_dataset.upper()],
                                        backbone_out_dim=in_dim,
                                        backbone=copy.deepcopy(backbone),
                                        preprocess= preprocess_transfer,
                                        freeze_backbone=False,
                                        freeze_backbone_bn=False).cuda()
            transfer_results_full = apply_transfer(args.transfer_dataset,
                                                logdir,
                                                transfer_args,
                                                transfer_model,
                                                transfer_ds,
                                                num_classes=datasets.DS_TO_NCLASSES[args.transfer_dataset.upper()],
                                                granularity=args.transfer_granularity)
        else:
            transfer_results_full = {}

        repo = git.Repo(search_parent_directories=True)
        args_dict = vars(args)
        args_dict['commit'] = repo.head.object.hexsha

        # Aggregate results
        out['transfer_datasets'][args.transfer_dataset] = {
            'args': args_dict,
            'transfer_results': transfer_results,
            'transfer_results_full': transfer_results_full,
        }

    output_pkl_file = os.path.join(args.output_pkl_dir, EXP_NAME + '.pt')
    torch.save(out, output_pkl_file)

    print("==>[Job successfully done.]")
