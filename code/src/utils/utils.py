import json
import random
import numpy as np
import torch
import logging, os, sys


def json_load(p):
    with open(p, "r") as fi:
        d = json.load(fi)
    return d


def kp3d_to_kp2d(kp3d, K):
    """
    Pinhole camera model projection
    K: camera intrinsics (3 x 3)
    kp3d: 3D coordinates wrt to camera (n_kp x 3)
    """
    kp2d = (kp3d @ K.T) / kp3d[..., 2:3]

    return kp2d[..., :2]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id, main_seed):
    seed = worker_id + main_seed
    set_seed(seed)


def pyt2np(tensor):
    return tensor.detach().cpu().numpy()


def add_logging(name, file, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count