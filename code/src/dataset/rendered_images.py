import os
import cv2 as cv
import glob
import numpy as np

from src.dataset.dataset_reader import DatasetReader


class RenderedImg(DatasetReader):
    def __init__(self, bz, transform):
        dataset_name = "RenderedImg"
        dataset_path = '/home/crl-5/Desktop/cloth_recon/tmp'
        self.bz = bz
        super().__init__(dataset_name, dataset_path, transform)

    def load_sample(self, idx):
        gt_path = os.path.join(self.dataset_path, "gt_%05d.png0.png" % idx)
        pred_path = os.path.join(self.dataset_path, "pred_%05d.png0.png" % idx)
        gt_img = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
        pred_img = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
        return {"gt_img": gt_img, "pred_img": pred_img}

    def __len__(self):
        x = len(glob.glob(self.dataset_path + '/*.png'))
        return x//2
