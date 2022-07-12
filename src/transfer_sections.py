from fastargs import Param, Section
from fastargs.validation import And, OneOf

def add_transfer_args():
    Section("transfer_configs", "config files location").params(
        aircraft=Param(str, "aircraft config file"),
        birdsnap=Param(str, "birdsnap config file"),
        caltech101=Param(str, "caltech101 config file"),
        caltech256=Param(str, "caltech256 config file"),
        chestxray14=Param(str, "chestxray14 config file"),
        cifar10=Param(str, "cifar10 config file"),
        cifar100=Param(str, "cifar100 config file"),
        cifar10_0_1=Param(str, "cifar10_0_1 config file"),
        cifar10_0_25=Param(str, "cifar10_0_25 config file"),
        cifar10_0_5=Param(str, "cifar10_0_5 config file"),
        flowers=Param(str, "flowers config file"),
        food=Param(str, "food config file"),
        imagenet=Param(str, "imagenet config file"),
        pets=Param(str, "pets config file"),
        stanford_cars=Param(str, "stanford_cars config file"),
        SUN397=Param(str, "SUN397 config file"),
    )

    Section("transfer_data", "transfer dataset").params(
        transfer_dataset=Param(str, "transfer dataset name"),
        transfer_path_train=Param(str, "transfer train loader"),
        transfer_path_val=Param(str, "transfer val loader"),
    )

    Section("transfer_training", "transfer training arguments").params(
        transfer_ffcv=Param(bool, "use ffcv transfer loaders"),
        upsample=Param(bool, "upsample input"),
        decoder_train=Param(str, "FFCV decoder for train set of target task"),
        decoder_val=Param(str, "FFCV decoder for val set of target task"),
        transfer_epochs=Param(int, "number of transfer epochs"),
        transfer_lr=Param(float, "transfer learning rate"),
        transfer_weight_decay=Param(float, "transfer weight decay"),
        transfer_momentum=Param(float, "transfer SGD momentum"),
        transfer_lr_scheduler=Param(And(str, OneOf(["steplr", "multisteplr", "cyclic"])), "transfer learning scheduler"),
        transfer_step_size=Param(int, "transfer step size"),
        transfer_lr_peak_epoch=Param(int, "transfer lr_peak_epoch for cyclic lr schedular"),
        transfer_gamma=Param(float, "tranfer SGD gamma"),
        transfer_label_smoothing=Param(float, "tranfer label smooothing"),
        transfer_lr_milestones=Param(str, "transfer learning rate milestones"),
        transfer_granularity=Param(And(str, OneOf(["global", "per_class"])), "Transfer Accuracy: global vs per class."),
        transfer_eval_epochs=Param(int, "Evaluate every n epochs."),
    )
