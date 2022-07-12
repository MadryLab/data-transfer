from argparse import ArgumentParser

from fastargs import Param, Section, get_current_config
from fastargs.decorators import param, section
from fastargs.validation import And, OneOf
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, ImageFolder

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', required=True),
    split=Param(And(str, OneOf(['train', 'val'])), 'Train or val set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length', required=True),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1)
)

@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
def main(dataset, split, data_dir, write_path, max_resolution, num_workers, chunk_size, subset,
jpeg_quality, write_mode):
    if dataset == 'cifar':
        my_dataset = CIFAR10(root=data_dir, train=(split == 'train'), download=True)
    elif dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    if subset > 0: my_dataset = Subset(my_dataset, range(subset))

    writer = DatasetWriter(len(my_dataset), write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=0.,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    })

    with writer:
        writer.write_pytorch_dataset(my_dataset, num_workers=num_workers, chunksize=chunk_size)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
