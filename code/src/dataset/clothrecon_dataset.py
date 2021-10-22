import os
import cv2 as cv

import numpy as np

from src.utils.utils import json_load
from src.dataset.dataset_reader import DatasetReader


class ClothReconDataset(DatasetReader):
    def __init__(self, split, data_transforms, dataset_path):

        # NOTE You may want to extend functionality such that you train on both
        # validation and training data for the final performance
        dataset_name = "ClothRecon"
        super().__init__(dataset_name, dataset_path, data_transforms)

        # train:val = 7:3
        # train_size = 8412
        # val_size = 3605

        self.split = split


    def load_sample(self, idx):
        img_path = os.path.join(self.dataset_path, f"{self.split}", "%05d.png" % idx)
        verts_path = os.path.join(self.dataset_path, f"{self.split}", "%05d.txt" % idx)
        # kp3d = self.kp3d[idx]
        # K = self.K[idx]
        # Load image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        verts = np.loadtxt(verts_path)

        return {"image": img, "verts": verts}

    def __len__(self):
        if self.split == 'val':
            return 3605
        elif self.split == 'train':
            return 8412
        else:
            raise NotImplementedError("ERROR in dataset split")
