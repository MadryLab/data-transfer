import ffcv.fields.decoders as decoders
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate
# These take in the desired image size, and the beton image size

IMAGE_DECODERS = {
    'simple': lambda imgsz: decoders.SimpleRGBImageDecoder(),
    'resized_crop': lambda imgsz: decoders.ResizedCropRGBImageDecoder((imgsz, imgsz)),
    'random_resized_crop': lambda imgsz: decoders.RandomResizedCropRGBImageDecoder((imgsz, imgsz)),
    'center_crop_256': lambda imgsz: decoders.CenterCropRGBImageDecoder((imgsz, imgsz), 224/256),
}

def cifar_train_aug(mean):
    return [
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, mean))),
        Cutout(4, tuple(map(int, mean))),
    ]

def immagenet_train_aug(hparams):
    return [RandomHorizontalFlip()]

IMAGE_AUGS = {
    'cifar': cifar_train_aug,
    'imagenet': immagenet_train_aug, 
}