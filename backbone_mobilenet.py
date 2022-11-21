from enum import Enum
from typing import Dict, List, Optional

import torch
import torchvision.models as models
from torch import nn
from torchvision.models import mobilenet
from torchvision.models import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork



class MobileNetBackbones(Enum):
    MOBILENET = "mobilenet"
    MOBILENETV2 = "mobilenet_v2"
    
    

    
class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(BackboneWithFPN, self).__init__()

        self.body = IntermediateLayerGetter(model=backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    
    
    

def get_mobilenet_backbone(backbone_name: MobileNetBackbones) -> torch.nn.Sequential:
    """
    Returns a mobilenet backbone pretrained on ImageNet.
    Removes the average-pooling layer and the linear layer at the end.
    """
    pretrained_model, out_channels = None, None

    if backbone_name == MobileNetBackbones.MOBILENET:
        pretrained_model = models.mobilenet(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == MobileNetBackbones.MOBILENETV2:
        pretrained_model = models.mobilenet_v2(pretrained=True, progress=False)
        out_channels = 3

    if not pretrained_model and not out_channels:
        raise ValueError(
            f"Your backbone_name is {backbone_name}, "
            f"but should be one of the following:"
            f"{[i.name for i in list(MobileNetBackbones)]}"
        )

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone




def get_mobilenet_fpn_backbone(
    backbone_name: MobileNetBackbones, pretrained: bool = True, trainable_layers: int = 5
) -> BackboneWithFPN:
    """
    Returns a resnet backbone with fpn pretrained on ImageNet.
    """
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=trainable_layers,
    )

    backbone.out_channels = 256
    return backbone




def mobilenet_fpn_backbone(
    backbone_name: MobileNetBackbones,
    pretrained: bool
):
    
    backbone = resnet.__dict__[backbone_name.value](
        pretrained=pretrained, norm_layer=norm_layer
    )

    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_list = [
       24, 32, 64, 1280
    ]
    
    out_channels = 256
    return BackboneWithFPN(
        backbone=backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
    )