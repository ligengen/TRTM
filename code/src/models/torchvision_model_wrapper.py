"""
Class to that interfaces with various torchvision models
"""
import torch
from torch import nn
import torchvision.models as models
from itertools import accumulate


def model_list():
    model_list = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    return model_list


class ResNetWrapper(nn.Module):
    """
    This is a resnet wrapper class which takes existing resnet architectures and
    adds a final linear layer at the end, ensuring proper output dimensionality
    """

    def __init__(self, model_name, output_slices):
        super().__init__()

        # Use a resnet-style backend
        if "resnet18" == model_name:
            model_func = models.resnet18
        elif "resnet34" == model_name:
            model_func = models.resnet34
        elif "resnet50" == model_name:
            model_func = models.resnet50
        elif "resnet101" == model_name:
            model_func = models.resnet101
        elif "resnet152" == model_name:
            model_func = models.resnet152
        else:
            raise Exception(f"Unknown backend model type: {model_name}")

        # Prepare the slicing
        slice_keys = list(output_slices.keys())
        slice_vals = list(output_slices.values())
        cumsum = list(accumulate(slice_vals))  # Hehe funneh name lolz
        # (key, start_idx, to_idx)
        output_idx = list(zip(slice_keys, [0] + cumsum[:-1], cumsum))
        # Construct the encoder.
        # NOTE You may want to look at the arguments of the resnet constructor
        # to test out various things:
        # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        b_model = model_func(pretrained=True)
        encoder = nn.Sequential(
            b_model.conv1,
            b_model.bn1,
            b_model.relu,
            b_model.maxpool,
            b_model.layer1,
            b_model.layer2,
            b_model.layer3,
            b_model.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        # Construct the final layer
        feat_dim = b_model.fc.in_features
        output_dim = cumsum[-1]
        final_layer = nn.Sequential(nn.Linear(feat_dim, output_dim))

        self.encoder = encoder
        self.final_layer = final_layer
        self.output_idx = output_idx

    def forward(self, x):
        # Get feature output
        f = self.encoder(x)
        # Get final output
        f = f.flatten(start_dim=1)
        out = self.final_layer(f)
        # Slice the output
        out_dict = {}
        for key, start_idx, to_idx in self.output_idx:
            out_dict[key] = out[..., start_idx:to_idx]

        return out_dict


def get_model(cfg_model):
    model = ResNetWrapper(
        model_name=cfg_model.name, output_slices=cfg_model.output_slices
    )

    return model
