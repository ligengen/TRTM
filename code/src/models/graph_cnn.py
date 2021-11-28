"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear
from .resnet import resnet50

class GraphCNN(nn.Module):
    
    def __init__(self, A, num_layers=5, num_channels=256):
        super(GraphCNN, self).__init__()
        self.A = A
        # TODO: add kp_info on ref_vert!
        self.ref_vertices = np.loadtxt('/home/crl-5/Desktop/cloth_recon/flat_state_050.txt') 
        self.ref_vertices = torch.from_numpy(self.ref_vertices).float().cuda()
        # self.kps = []
        # for i in range(len(self.template_mesh_verts)):
        #     if abs(self.template_mesh_verts[i][0]) == 1 or abs(self.template_mesh_verts[i][1]) == 1:
        #         self.kps.append(i)
        # kp_info = np.zeros((world_pos.shape[0], 1))
        # kp_info[self.kps, :] = 1.0
        self.resnet = resnet50(pretrained=False)
        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        # self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                               nn.ReLU(inplace=True),
        #                               GraphLinear(num_channels, 1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(A.shape[0], 3))

    def forward(self, image):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        image_resnet = self.resnet(image)
        image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        # camera = self.camera_fc(x).view(batch_size, 3)
        # return shape, camera
        return shape
