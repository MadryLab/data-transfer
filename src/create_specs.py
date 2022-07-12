import json
import os

from fastargs import Param, Section
from fastargs.decorators import param
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from tqdm import tqdm

from ffcv_utils import get_ffcv_loader
from memmap_utils import make_config
from spec_utils import get_schema, transfer_datasets

Section('data').params(
    root=Param(str, 'ffcv data loaders path', required=True),
    specs_dir=Param(str, 'directory to create specs in', default='specs'),
)

@param('data.root')
@param('data.specs_dir')
def main(root, specs_dir):

    if os.path.exists(specs_dir):
        raise Exception(f'The folder "{specs_dir}" exist. Please delete the folder if you want to overwrite it')
    else:
        os.makedirs(specs_dir)

    for tr_ds in tqdm(transfer_datasets):
        if "cifar10_" in tr_ds:
            tr_ds_path = os.path.join(root, f"{tr_ds}/cifar10_test.beton")
        else:
            tr_ds_path = os.path.join(root, f"{tr_ds}/{tr_ds}_test.beton")

        img_decoder = RandomResizedCropRGBImageDecoder((32,32))
        tr_loader = get_ffcv_loader(split="test", batch_size=32, ds_name=tr_ds, img_decoder=img_decoder, drop_last=False, val_path=tr_ds_path, num_workers=1)
        num_samples = tr_loader.reader.num_samples

        tr_scheme = get_schema(test_size=int(num_samples))

        if tr_ds == "chestxray14":
            tr_scheme["schema"]["auroc"] = {
                "dtype": "float32",
                "shape": []
            }

        tr_scheme_path = os.path.join(specs_dir, f"{tr_ds}.json")
        with open(tr_scheme_path, "w") as f:
            json.dump(tr_scheme, f, indent=4)

if __name__ == '__main__':
    make_config()
    main()
    print('Done!')
