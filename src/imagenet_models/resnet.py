import torch
from torch.nn.modules.dropout import Dropout
from torchvision import transforms
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
ch = torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            separable=False, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #print('3x3', in_planes, out_planes)
    if not separable:
        conver = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)
    if separable:
        raise ValueError('no')
        # conver = depthwise_separable_conv(in_planes, out_planes, 3, dilation, dilation=dilation, stride=stride)
    return conver


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    #print('1x1', in_planes, out_planes)
    conver = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return conver #ConvSandwich(conver)

from functools import partial

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type='3x3',
        separable=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conver = partial(conv3x3, separable=separable) if conv_type == '3x3' else conv1x1
        self.conv1 = conver(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conver(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


import torch as ch
#print('VERSION', ch.__version__)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# from galactic_chungus import BiasedConv
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        initial_filt=7,
        initial_dilation=1,
        initial_stride=2,
        maxpool_size=3,
        maxpool_stride=2,
        first_layer_stride=1,
        second_layer_stride=2,
        fourth_layer_stride=2,
        tta=True,
        pooltype='max',
        final_layer_conv_type='3x3',
        separable_firstlayer=False,
        separable_3x3s=False,
        bss=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.tta = tta
        self.inplanes = 64 if not bss else bss[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if not separable_firstlayer:
            #print('first', 3, self.inplanes)
            conver = nn.Conv2d(3, self.inplanes, kernel_size=initial_filt,
                               stride=initial_stride, padding=3, bias=False)
        else:
            raise ValueError('no')

        self.conv1 = conver #BiasedConv(conver, mode='fast')
        # self.conv1 = ConvSandwich(conver)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if pooltype == 'max':
            self.maxpool = nn.MaxPool2d(kernel_size=maxpool_size, stride=maxpool_stride, padding=1)
        elif pooltype == 'avg':
            self.maxpool = nn.AvgPool2d(kernel_size=maxpool_size, stride=maxpool_stride, padding=1)
        else:
            raise ValueError('chungus')

        bss = bss or [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, bss[0], layers[0], stride=first_layer_stride,
                                       separable=False)
        self.layer2 = self._make_layer(block, bss[1], layers[1], stride=second_layer_stride,
                                       dilate=replace_stride_with_dilation[0],
                                       separable=separable_3x3s)
        self.layer3 = self._make_layer(block, bss[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, bss[3], layers[3], stride=fourth_layer_stride,
                                       dilate=replace_stride_with_dilation[2],
                                       conv_type=final_layer_conv_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(bss[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, **kwargs) -> nn.Sequential:
        assert not dilate, 'Uh oh.'
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            ## RESNET-D
            # pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False)
            downsample = nn.Sequential(
                # pool,
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # ops = [
        #     self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
        #     self.layer2, self.layer3, self.layer4, self.avgpool,
        #     partial(torch.flatten, start_dim=1), self.fc
        # ]
        # names = [
        #     'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3',
        #     'layer4', 'avgpool', 'flatten', 'fc'
        # ]

        # total_time = 0
        # for ii, op in zip(names, ops):
        #     # start = time.time()
        #     # if ii == 'bn1':
        #     #     x = x.contiguous()

        #     # #print(ii)
        #     x = op(x)
        #     # delta = time.time() - start
        #     # total_time += delta
        #     # #print(ii, delta)
        # #print('total', total_time)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.training or not self.tta:
            return self._forward_impl(x)
        return self._forward_impl(x) + self._forward_impl(ch.flip(x, dims=[3]))

import time

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError('sorry u cant do that')
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet18_custom(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert 'bss' in kwargs and 'initial_filt' in kwargs
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet18_smallfilt(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_smallfilt', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   initial_filt=5, **kwargs)

def resnet18_smallmaxpool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_smallmaxpool', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=2, **kwargs)

def resnet18_smallmaxpoolsmallfilt(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_smallmaxpoolsmallfilt', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=2, initial_filt=5, **kwargs)

def resnet18_dilated(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_dilated', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   initial_dilation=2, initial_stride=4, **kwargs)

def resnet18_largefilt(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_largefilt', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   initial_filt=13, initial_stride=4, **kwargs)

def resnet18_bigmaxpool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_bigmaxpool', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=6, **kwargs)

def resnet18_bigmaxpoolfixed(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_bigmaxpoolfixed', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=6, maxpool_stride=4, **kwargs)

def resnet18_smallmaxpoolbigstride(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_smallmaxpoolbigstride', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=3, maxpool_stride=4, **kwargs)

def resnet18_bigavgpool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_bigavgpool', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   maxpool_size=6, maxpool_stride=4, pooltype='avg', **kwargs)

def resnet18_extradownsize(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_extradownsize', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, first_layer_stride=2, **kwargs)

def resnet18_nodownsize(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_nodownsize', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, second_layer_stride=1, **kwargs)

def resnet18_onexone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_onexone', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, final_layer_conv_type='1x1', fourth_layer_stride=1, **kwargs)

def resnet18_onexone_betterstem(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_onexone', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, final_layer_conv_type='1x1', fourth_layer_stride=1,
                   maxpool_size=2, initial_filt=5, **kwargs)

def resnet18_onexone_betterstem2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18_onexone', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, final_layer_conv_type='1x1', fourth_layer_stride=1,
                   maxpool_size=2, initial_filt=3, **kwargs)


def resnet18_bilinear(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_bilinear', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, **kwargs)


def resnet18_sep1(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_sep1', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, separable_firstlayer=True, **kwargs)

def resnet18_sep1_all(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_sep1_all', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, separable_firstlayer=True, separable_3x3s=True, **kwargs)

def resnet18_sep0_all(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet('resnet18_sep0_all', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, separable_3x3s=True, **kwargs)

import numpy as np
def wide_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([64, 128, 256, 512]) * 1.5))
    return _resnet('wide_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def narrow_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([64, 128, 256, 512]) / 1.5))
    return _resnet('narrow_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def narrower_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([64, 128, 256, 512]) / 2))
    return _resnet('narrow_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def narrowish_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([64, 128, 256, 512]) / 1.25))
    return _resnet('narrow_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def narrow_neck_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([32, 64, 256, 512])))
    return _resnet('narrow_neck_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def narrower_neck_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    bss = list(map(int, np.array([20, 40, 256, 512])))
    return _resnet('narrower_neck_resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, bss=bss, **kwargs)

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def make_imagenet_model(model_class):
    if model_class == 'linear_inc_imagenet' or model_class == 'imagenet_sd':
        neck_fracs = [0.25, 0.5, 1, 2]
    elif model_class == 'relu_inc_imagenet':
        neck_fracs = [0.25, 0.5, 1, 1]
    elif model_class == 'imagenet':
        neck_fracs = [1, 1, 1, 1]
    else:
        neck_fracs = [1, 1, 1, 1]

    model_args = {
        'bss': list(map(int, np.array([
            64 * neck_fracs[0],
            128 * neck_fracs[1],
            256 * neck_fracs[2],
            512 * neck_fracs[3]]))),
        'initial_filt': 7
    }

    model = resnet18_custom(**model_args)
    return model