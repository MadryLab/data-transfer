import argparse
from types import SimpleNamespace

import yaml
from fastargs import get_current_config


def process_args_and_config():
    config = get_current_config()
    parser = argparse.ArgumentParser(description="Data Transfer")
    # parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    # args = parser.parse_args()

    # config.collect_config_file(args.config)
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()

    config_args = config.get()
    transfer_configs = vars(config_args.transfer_configs)
    del config_args.transfer_configs

    args = convert_fastargs(config_args)
    args = SimpleNamespace(**args)
    
    if hasattr(args, "lr_milestones"):
        convert_arg_to_list(args, ["lr_milestones"])
    if hasattr(args, "exclude_file"):
        convert_emptystr_to_None(args, ["exclude_file"])
    return args, transfer_configs

def convert_fastargs(fast_args):
    args_dict = {}
    fast_args_vars = vars(fast_args)

    for key, value in fast_args_vars.items():
        if isinstance(value, SimpleNamespace):
            args_dict.update(convert_fastargs(value))
        else:
            args_dict[key] = value

    return args_dict

def convert_arg_to_list(args, keys):
    for key in keys:
        if getattr(args, key) == "":
            setattr(args, key, [])
        else:
            val_list = [int(x) for x in getattr(args, key).split(",")]
            setattr(args, key, val_list)

def convert_emptystr_to_None(args, keys):
    for key in keys:
        if getattr(args, key) == "":
            setattr(args, key, None)

def read_yaml(yaml_dir):
    with open(yaml_dir, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file

def update_tr_args(args, tr_args):
    args = vars(args)
    for key, value in tr_args.items():
        if isinstance(value, dict):
            args.update(value)

    args = SimpleNamespace(**args)
    convert_arg_to_list(args, ["transfer_lr_milestones"])

    return args

def update_config_with_transfer_config(transfer_config_path):
    config = get_current_config()
    config.collect_config_file(transfer_config_path)
    config.validate(mode='stderr')
    config.summary()
    config_args = config.get()

    args = convert_fastargs(config_args)
    args = SimpleNamespace(**args)
    convert_arg_to_list(args, ["transfer_lr_milestones"])
    return args
