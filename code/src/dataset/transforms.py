import torch
import cv2 as cv  # cv is faster than PIL

import numpy as np
import random
import torchvision.transforms as transforms
from src.utils.utils import kp3d_to_kp2d
from skimage.util import random_noise


class NpToPytSingleChannel:
    def __call__(self, sample):
        sample['gt_img'] = sample['gt_img'].astype(np.float32) / 255
        sample['pred_img'] = sample['pred_img'].astype(np.float32) / 255
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v).float()
        return sample


class NumpyToPytorch:
    def __call__(self, sample):
        # Torch take C x H x W whereas np,cv use H x W x C
        img = sample["image"].transpose(2, 0, 1)
        # For single channel img, should transpose or leave it alone?
        sample["image"] = img.astype(np.float32) / 255
        # Transfrom from numpy array to pytorch tensor
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v).float()

        return sample


class Resize:
    """
    Resizes the image to img_size
    """

    def __init__(self, img_size):
        self.img_size = tuple(img_size)

    def __call__(self, sample):
        sample["image"] = cv.resize(sample["image"], self.img_size)
        # sample['image'] = np.expand_dims(sample['image'], 2)
        return sample


class Occlusion:
    def __init__(self):
        pass

    def __call__(self, sample):
        # image occlusion
        depth_img = sample['image']
        occlusion_num = np.random.randint(5, 9)
        for i in range(occlusion_num):
            x = np.random.randint(100, 980)
            y = np.random.randint(100, 980)
            r = np.random.randint(40, 150)
            a1 = random.random() * 2 * np.pi
            while True:
                a2 = random.random() * 2 * np.pi
                if abs(a1-a2) >= np.pi*0.15 and abs(a1-a2) <= np.pi*0.75:
                    break
            
            x1 = int(x + r * np.cos(a1))
            y1 = int(y + r * np.sin(a1))
            x3 = int(x + r * np.cos(a1 + np.pi))
            y3 = int(y + r * np.sin(a1 + np.pi))

            x2 = int(x + r * np.cos(a2))
            y2 = int(y + r * np.sin(a2))
            x4 = int(x + r * np.cos(a2 + np.pi))
            y4 = int(y + r * np.sin(a2 + np.pi))

            contours = np.array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]], dtype=np.int32)
            cv.fillPoly(depth_img, contours, (255., 255., 255.))

        sample['image'] = depth_img
        return sample


class Augmentation:
    def __init__(self):
        # self.noise_factor = 0.2
        self.scale_factor = 0.2
        self.rot_factor = 90
        # self.pn = np.ones(3) 
        # picture noise should be the same across the channels!
        # self.pn = 1.0
        # self.rot = rot
        # self.sc = sc


    def __call__(self, sample):
        # if self.rot is None:
        # self.sc = 1.0
        # self.pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor)
        # self.pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor)
        self.rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
        self.sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
        if np.random.uniform() <= 0.6:
            self.rot = 0
        if np.random.uniform() <= 0.5:
            self.sc = 1.0
        
        # TODO no scaling at all
        self.sc = 1.0

        depth_img = sample['image']
        # image roration
        M = cv.getRotationMatrix2D([540, 540], self.rot, self.sc)
        depth_img = cv.warpAffine(depth_img, M, (1080, 1080), borderValue=(255.,255.,255.))
        # image noise
        # TODO
        # if np.random.uniform() <= 0.5:
        #     depth_img = random_noise(depth_img, var=0.0005)
        #     depth_img = np.array(255 * depth_img).astype(np.uint8)

        # depth_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,0]*self.pn[0]))
        # depth_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,0]*self.pn))
        # depth_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,1]*self.pn[1]))
        # depth_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,1]*self.pn))
        # depth_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,2]*self.pn[2]))
        # depth_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, depth_img[:,:,2]*self.pn))

        sample['image'] = depth_img

        # kp3d = sample['kp3d']
        rot_mat = np.eye(3)
        if not self.rot == 0:
            rot_rad = self.rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        # kp3d = np.einsum('ij,kj->ki', rot_mat, kp3d)
        # sample['kp3d'] = kp3d

        verts = sample['verts']
        verts = np.einsum('ij,kj->ki', rot_mat, verts)
        sample['verts'] = verts

        return sample


'''class ScaleNormalize:
    """
    Scale normalizes the 3D joint position by the MCP bone of the index finger.
    The resulting 3D joint skeleton has an index MCP bone length of 1
    NOTE: This function will throw a warning for the test data, as the ground-truth
    is set to 0. This is because they are not available.
    """

    def __init__(self):
        # NOTE Try taking the bone as parameter and see if scale normalizing other bones
        # affects performance
        pass

    def __call__(self, sample):
        kp3d = sample["kp3d"]
        bone_length = np.linalg.norm(
            kp3d[JointInfo.index_mcp] - kp3d[JointInfo.index_pip]
        )
        kp3d = kp3d / bone_length
        sample["kp3d"] = kp3d

        return sample
'''

