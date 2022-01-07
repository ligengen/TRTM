import os
import pdb
import glob
import cv2 as cv

import numpy as np

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
        # img_path = os.path.join("/home/crl-5/Desktop/occlusion", "%05d.png" % idx)
        verts_path = os.path.join(self.dataset_path, f"{self.split}", "%05d.txt" % idx)
        # verts_path = os.path.join("/home/crl-5/Desktop/occlusion", "%05d.txt" % idx)
        # kp3d = self.kp3d[idx]
        # K = self.K[idx]
        # Load image
        # img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = cv.imread(img_path)
        # img = np.expand_dims(img, axis=2)
        verts = np.loadtxt(verts_path)

        return {"image": img, "verts": verts}

    def __len__(self):
        if self.split == 'val':
            return len(glob.glob(os.path.join(self.dataset_path, "val", "*.txt")))
        elif self.split == 'train':
            return len(glob.glob(os.path.join(self.dataset_path, "train", "*.txt")))
            # return 8412
        elif self.split == 'test':
            return len(glob.glob(os.path.join(self.dataset_path, "test", "*.txt")))
            # return 4303
        else:
            raise NotImplementedError("ERROR in dataset split")


if __name__ == '__main__':
    from src.dataset import transforms_factory
    from easydict import EasyDict as edict
    import cv2
    
    dataset_path = '/home/crl-5/Desktop/50'
    split = 'test'
    transformation_cfg = edict({"Augmentation": {}})
    transformations = transforms_factory.get_transforms(transformation_cfg)
    dataset = ClothReconDataset(split, transformations, dataset_path)

    idx = 13
    sample = dataset.load_sample(idx)
    sample = transformations(sample)
    img = sample['image']
    verts = sample['verts']
    cv2.imwrite('/home/crl-5/Desktop/occlusion/%05d.png'%idx, img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # np.savetxt('/home/crl-5/Desktop/test.txt', verts)

