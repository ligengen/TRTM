import inspect
from torchvision.transforms import Compose
import src.dataset.transforms

# Get all transformations defined in src.dataset.transforms
transform_map = {
    t[0]: t[1] for t in inspect.getmembers(src.dataset.transforms, inspect.isclass)
}


def get_transforms(transform_cfg):
    """ Returns a composite of data transformations as defined by transform_cfg
    """
    t = []
    for t_type, t_param in transform_cfg.items():
        t.append(transform_map[t_type](**t_param))

    # t.append(transform_map["NumpyToPytorch"]())  # We always need to convert to pytorch
    t = Compose(t)

    return t
