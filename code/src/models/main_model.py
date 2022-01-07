import torch
import torch.nn as nn

from src.models import model_factory

class MainModel(nn.Module):
    """
    This is a bare bones main model class. This is where you abstract away the exact
    backbone architecture detail and do fancy stuff with the output of the backbone.
    E.g for MANO you may want to construct the MANO mesh based on the parameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone_model = model_factory.get_model(cfg.backend)
        # self.backend_model = model_factory.get_model(cfg.backend)
        self.cloth_model = model_factory.get_model(cfg.cloth_model)

    def forward(self, batch, is_training, read_intermediate, offset):
        image_feature = self.backbone_model(batch)
        # output = self.backend_model(batch)
        node_pred = self.cloth_model(image_feature, is_training, read_intermediate, offset)
        # Adjust shape of output
        # output["verts"] = output["verts"].view(-1, 2601, 3)
        # return image_feature.view(-1, 2601, 3)
        # return output['verts']
        return node_pred
