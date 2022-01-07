import torch

from src.models import torchvision_model_wrapper
from src.models import main_model
from src.models import cloth_model

def get_model(cfg_model, dev=None, logger=None, is_eval=None, ckpt=None):
    model_name = cfg_model.name.lower()

    if dev is None:
        dev = torch.device("cpu")

    if model_name == "main_model":
        model = main_model.MainModel(cfg_model)
        if is_eval:
            logger.info("Evaluation: Loading from checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt)['model_state_dict'])
    elif model_name in torchvision_model_wrapper.model_list():
        # ResNet
        model = torchvision_model_wrapper.get_model(cfg_model)
    elif model_name == 'cloth_model':
        model = cloth_model.Model()
    else:
        # NOTE You may want to test other model types which are not supported by resnet_wrapper
        raise Exception(f"Unsupported model {model_name}")

    # Push to desire device (cpu/gpu)
    model.to(dev)

    return model
