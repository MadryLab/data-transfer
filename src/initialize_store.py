import json
import os

import numpy as np
from fastargs import Param, Section
from fastargs.decorators import param
from tqdm import tqdm

from memmap_utils import make_config
from spec_utils import preprocess_spec, transfer_datasets

Section('logging').params(
    logdir=Param(str, 'where to log memmaps', required=True),
)

@param('logging.logdir')
def main(logdir):
    print("Logging in: ", logdir)

    if os.path.exists(logdir):
        raise Exception(f'The folder "{logdir}" exist. Please delete the folder if you want to overwrite it')
    else:
        os.makedirs(logdir)

    tqdm_iter = tqdm(transfer_datasets, "Creating transfer memmaps...")
    for tr_dataset in tqdm_iter:
        spec_path = os.path.join("specs", f"{tr_dataset}.json")

        spec_file = json.loads(open(spec_path, 'r').read())
        spec_file = preprocess_spec(spec_file)

        num_models = spec_file["num_models"]
        for key, metadata in spec_file["schema"].items():
            dtype = getattr(np, metadata['dtype'])
            shape = (num_models,) + tuple(metadata['shape'])

            for tr_type in ["full", "fixed"]:
                tr_logdir = os.path.join(logdir, tr_dataset, tr_type)
                os.makedirs(tr_logdir, exist_ok=True)
                this_filename = os.path.join(tr_logdir, f'{key}.npy')
                mmap = np.lib.format.open_memmap(this_filename, mode='w+', dtype=dtype,
                                                shape=shape)
                mmap.flush()

if __name__ == '__main__':
    make_config()
    main()
    print('Done!')
