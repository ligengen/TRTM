import torch
from torch import nn
from src.models import gat
import torch.nn.functional as F
import numpy as np
import pdb


class Model(nn.Module):
    def __init__(self, message_passing_steps=15, name='Model'):
        super(Model, self).__init__()
        self.template_mesh_verts = np.loadtxt('/home/crl-5/Desktop/cloth_recon/flat_state_050.txt')
        self.edge_idx = np.loadtxt('/home/crl-5/Desktop/cloth_recon/mesh_edge_idx.txt').astype(int)
        self.message_passing_steps = message_passing_steps
        self.learned_model = gat.GAT(
                output_size=3, #TODO: graph_feat or node_pred?
                latent_size=128, #128
                num_layers=2,
                message_passing_steps=self.message_passing_steps)
       
    def _build_graph(self, is_training):
        """Builds template-input graph."""
        world_pos = torch.from_numpy(self.template_mesh_verts).float().cuda()
        # template_verts = torch.unsqueeze(world_pos, 0)
        # template_verts = world_pos # do not set batch_size because everything is the same!
        self.kps = []
        for i in range(len(self.template_mesh_verts)):
            if abs(self.template_mesh_verts[i][0]) == 1 or abs(self.template_mesh_verts[i][1]) == 1:
                self.kps.append(i)
        # kp_info = np.zeros((template_verts.shape[0], template_verts.shape[1], 1))
        kp_info = np.zeros((world_pos.shape[0], 1))
        # kp_info[:, self.kps, :] = 1.0
        kp_info[self.kps, :] = 1.0
        kp_info = torch.from_numpy(kp_info).float().cuda()
        # node feature: 3d coord + 1d iskp
        # node_features = torch.cat([world_pos, kp_info], dim=-1)
        # TODO: removed keypoints feature!
        node_features = world_pos
        # node_features = node_features.expand(batch_size, -1, -1)  

        # create two-way connectivity
        senders = self.edge_idx[:, 0]
        receivers = self.edge_idx[:, 1]
        sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64).cuda()
        receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64).cuda()
        relative_world_pos = (torch.index_select(world_pos, 0, sender) - torch.index_select(world_pos, 0, receiver))
        # relative_world_pos = torch.unsqueeze(relative_world_pos, 0).expand(batch_size, -1, -1)
        relative_norm = torch.norm(relative_world_pos, dim=-1, keepdim=True)
        edge_features = torch.cat((relative_world_pos, relative_norm), -1)

        # receiver = torch.unsqueeze(receiver, 0).expand(batch_size, -1, -1)
        # sender = torch.unsqueeze(sender, 0).expand(batch_size, -1, -1)

        mesh_edges = gat.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receiver,
            senders=sender)


        return gat.MultiGraph(node_features=node_features, edge_sets=[mesh_edges])

    def forward(self, image_feature, is_training, read_intermediate, vis_att):
        # batch_size = image_feature.shape[0]
        # generate T-pose template mesh
        self.graph = self._build_graph(is_training)

        node_pred = self.learned_model(self.graph, image_feature, read_intermediate, vis_att)
        # feature = torch.cat((image_feat, node_pred), -1)
        return node_pred


