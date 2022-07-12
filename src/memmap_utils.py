import os
import types
from argparse import ArgumentParser

import numpy as np
from fastargs import get_current_config

from spec_utils import SCHEMA_ENTRIES


def collect_known_args(self, parser, disable_help=False):
    args, _ = parser.parse_known_args()
    for fname in args.config_file:
        self.collect_config_file(fname)

    args = vars(args)
    self.collect(args)
    self.collect_env_variables()

def make_config(quiet=False):
    config = get_current_config()
    f = types.MethodType(collect_known_args, config)
    config.collect_argparse_args = f

    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config

def memmap_read(k, logdir, index):
    this_filename = os.path.join(logdir, f"{k}.npy")
    # make sure that the mmap actually exists
    assert os.path.exists(this_filename)

    mmap = np.lib.format.open_memmap(this_filename, mode='r')
    return mmap[index]

def memmap_log(k, v, logdir, index):
    this_filename = os.path.join(logdir, f"{k}.npy")
    # make sure that the mmap actually exists
    assert os.path.exists(this_filename)

    # now acutally open it for writing
    mmap = np.lib.format.open_memmap(this_filename, mode='r+')
    mmap[index] = v
    mmap.flush()

def memmap_loop(results, mmap_logdir, index):
    for k, v in results.items():
        if k not in SCHEMA_ENTRIES:
            continue

        v = v.cpu().detach().numpy()
        memmap_log(k, v, mmap_logdir, index)
