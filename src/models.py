import os
import torch
import torch.nn as nn

class TransferNetwork(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone_out_dim,
                 backbone,
                 preprocess=None, # a preprocessing nn module (e.g, upsample for CIFAR). Note, preprocess will not be backpropped
                 freeze_backbone=False, # whether to freeze the backbone
                 freeze_backbone_bn=False, # whether to freeze the batchnorm in the backbone
                ):
        super().__init__()
        self.preprocess = preprocess
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_bn = freeze_backbone_bn
        self.fc = nn.Linear(backbone_out_dim, num_classes)

        if self.preprocess is not None: # freeze everything in preprocess
            for p in self.preprocess.parameters():
                p.requires_grad = False
        if self.freeze_backbone: #optionally freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, **fwd_args):
        if self.preprocess is not None:
            with torch.no_grad():
                self.preprocess = self.preprocess.eval()
                x = self.preprocess(x)

        # backbone
        if self.freeze_backbone_bn:
            self.backbone = self.backbone.eval()
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x, **fwd_args)
        else:
            x = self.backbone(x, **fwd_args)
        if len(x.shape)==4:
            x = x.squeeze(-1).squeeze(-1)
        assert len(x.shape) == 2
        # final layer
        return self.fc(x)

from imagenet_models.resnet import make_imagenet_model
from collections import OrderedDict
def get_imagenet_model(id_dir, checkpoint_dir, checkpoint_id):
    model = make_imagenet_model('linear_inc_imagenet')
    model_ids = torch.load(id_dir)
    model_id = model_ids[checkpoint_id]
    checkpoint_path = os.path.join(checkpoint_dir, str(model_id), '14_checkpoint.pt')

    checkpoint = torch.load(checkpoint_path)

    def rename_state_dict(state_dict):
        new_state_dict = OrderedDict()
        del state_dict['normalizer.new_mean']
        del state_dict['normalizer.new_std']

        def _transform_keys(k):
            if k == 'model.model.9.weight':
                return 'fc.0.weight'
            assert k[:5] == 'model'
            return k[6:]

        for key, value in state_dict.items():
            new_key = _transform_keys(key)
            new_state_dict[new_key] = value
        return new_state_dict

    model.load_state_dict(rename_state_dict(checkpoint['model']))
    return model