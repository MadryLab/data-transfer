# from wandb import Config
import copy
import os
import sys
import uuid

import git
import torch
import torch.nn as nn
from time import sleep
from functools import partial
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from tqdm.contrib.concurrent import thread_map

import config_parse_utils
import datasets
import ffcv_utils
from decoders_and_transforms import IMAGE_DECODERS
from ffcv_aug import SelectLabel
from memmap_utils import memmap_log, memmap_loop, memmap_read
from models import TransferNetwork, get_imagenet_model
from trainer import LightWeightTrainer
from transfer_sections import add_transfer_args
from eval_utils import evaluate_model

from ipdb import set_trace as bp
from IPython import embed

Section("training", "training arguments").params(
    num_workers=Param(int, "number of workers", default=8),
    batch_size=Param(int, "batch size", default=512),
    exp_name=Param(str, "experiment name", default=""),
    transfer_task=Param(And(str, OneOf(["ALL", "FIXED", "FULL"])), "transfer task", default="ALL"),
    disable_logging=Param(bool, "disable logging", default=False),
    supercloud=Param(bool, "use supercloud", default=False),
    data_root=Param(str, "data root dir", default="/mnt/nfs/datasets/transfer_datasets"),
    decoder=Param(str, "FFCV image decoder.", default='random_resized_crop'),
    granularity=Param(And(str, OneOf(["global", "per_class"])), "Accuracy: global vs per class.", default='global'),
    eval_epochs=Param(int, "Evaluate every n epochs.", default=5),
    transfer_dataset_override=Param(str, "Set in order to override the transfer dataset used", default=''),
)

Section("model", "model architecture arguments").params(
    arch=Param(str, "architecture to train", default="resnet18"),
    pretrained=Param(bool, "Pytorch Pretrained", default=False),
    id_dir=Param(str, "path of IN example-based model ids", default=""),
    checkpoint_dir=Param(str, "directory of IN example-based checkpoints", default=""),
    job_index=Param(int, "job index", default=0),
    models_per_job=Param(int, "number of models per job", default=1),
    max_workers=Param(int, "number of parallel jobs", default=1)
)

Section("logging", "memmaps logging dir").params(
    mmap_logdir=Param(str, "where to log memmap results", required=True),
    do_if_complete=Param(bool, "over-write previous results", default=False)
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
    val_path=Param(str, "path of validation loader", default="/home/gridsan/groups/robustness/datasets/data-transfer/imagenet-val-256.pxy"),
)

add_transfer_args()

def work(ix, args, transfer_configs, jid, models_per_job, max_workers):
    # try without? not sure why G had it
    if ix < max_workers:
        to_sleep = ix * 20
        sleep(to_sleep)

    model_index = jid * models_per_job + ix
    if model_index >= 71828: # this is hardcoded - clean later
        print(f"==>[No checkpoint for model {model_index}. Skipping.]")
        return

    print(f"==>[Job {model_index} started.]")

    transfer_dataset_override = args.transfer_dataset_override
    assert not args.pretrained

    EXP_NAME = str(uuid.uuid4()) if not args.exp_name else args.exp_name

    # load imagenet pretrained model
    model = get_imagenet_model(args.id_dir, args.checkpoint_dir, model_index)
    print(f"==>[Model checkpoint loaded.]")

    # common_args = {
    #     'ds_name': "imagenet",
    #     'batch_size': args.batch_size,
    #     'num_workers': args.num_workers,
    #     'val_path': os.path.join(args.data_root, args.val_path),
    #     'quasi_random': args.supercloud,
    #     'dataset_mean': datasets.DS_TO_MEAN['IMAGENET'],
    #     'dataset_std': datasets.DS_TO_STD['IMAGENET'],
    # }
    # val_loader = ffcv_utils.get_ffcv_loader(split='val',
    #                 img_decoder=IMAGE_DECODERS['center_crop_256'](224), **common_args)

    # evaluate_model(model, val_loader, num_classes=1000)

    # Extract backbone
    in_dim = model.fc.in_features
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    def apply_transfer(name, logdir, transfer_args, transfer_model, datamodule, num_classes, granularity):

        trainer = LightWeightTrainer(transfer_args, name, logdir, enable_logging=not args.disable_logging, granularity=granularity)
        trainer.fit(transfer_model, datamodule.train_dataloader(), datamodule.val_dataloader())
        return evaluate_model(transfer_model, datamodule.val_dataloader(), num_classes=num_classes, granularity=granularity, name=name)


    for _, transfer_config_path in transfer_configs.items():
        data_root = args.data_root
        args = config_parse_utils.update_config_with_transfer_config(transfer_config_path)
        if transfer_dataset_override and args.transfer_dataset != transfer_dataset_override:
            continue
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

        memmap_read_flags = [memmap_read("_completed", os.path.join(args.mmap_logdir, args.transfer_dataset.lower(), x), model_index) for x in ["fixed", "full"]]
        if (not args.do_if_complete) and all(memmap_read_flags):
            print('Already completed, skipping... '
              '(use the --logging.do_if_complete flag to override)')

            continue

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


            # Save the logs in memmaps
            memmap_logdir_fixed = os.path.join(args.mmap_logdir, args.transfer_dataset.lower(), "fixed")
            memmap_loop(transfer_results, memmap_logdir_fixed, model_index)
            memmap_log("_completed", 1, memmap_logdir_fixed, model_index)

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

            # Save the logs in memmaps
            memmap_logdir_full = os.path.join(args.mmap_logdir, args.transfer_dataset.lower(), "full")
            memmap_loop(transfer_results_full, memmap_logdir_full, model_index)
            memmap_log("_completed", 1, memmap_logdir_full, model_index)


        repo = git.Repo(search_parent_directories=True)
        args_dict = vars(args)
        args_dict['commit'] = repo.head.object.hexsha

        torch.save(args_dict, os.path.join(args.mmap_logdir, args.transfer_dataset.lower(), "args.pt"))

    with open(os.path.join(args.mmap_logdir, "cmds.txt"), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    print(f"==>[Job {model_index} successfully done.]")

if __name__ == "__main__":
    args, transfer_configs = config_parse_utils.process_args_and_config()

    jid=args.job_index
    max_workers=args.max_workers
    models_per_job=args.models_per_job

    parallel_work = partial(work,
                            args=copy.deepcopy(args),
                            transfer_configs=copy.deepcopy(transfer_configs),
                            jid=jid,
                            max_workers=max_workers,
                            models_per_job=models_per_job)


    # parallel_work(0) # for testing
    thread_map(parallel_work, range(models_per_job), max_workers=max_workers)
