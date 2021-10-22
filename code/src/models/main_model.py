import torch
import torch.nn as nn

from src.models import model_factory

# TODO! add meshcnn here
class MainModel(nn.Module):
    """
    This is a bare bones main model class. This is where you abstract away the exact
    backbone architecture detail and do fancy stuff with the output of the backbone.
    E.g for MANO you may want to construct the MANO mesh based on the parameters
    """

    def __init__(self, cfg):
        super().__init__()
        # NOTE You can try different backend models here
        self.backend_model = model_factory.get_model(cfg.backend)

    def forward(self, batch):
        # Feed through backend model
        output = self.backend_model(batch["image"])
        # Adjust shape of output
        output["kp3d"] = output["kp3d"].view(-1, 21, 3)

        return output
