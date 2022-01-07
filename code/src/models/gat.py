import collections
from collections import OrderedDict
import functools
import torch
from torch import nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import pdb

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super(LazyMLP, self).__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        y = self.layers(input)
        return y

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, idx):
        input = self.linear(input)
        input = self.activation(input)
        attention = torch.zeros(*input.shape)
        attention = scatter_softmax(input, idx, dim=1)
        return attention.cuda()

class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size):
        super(GraphNetBlock, self).__init__()
        self.edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.attention_model = Attention()

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        sender_features = torch.index_select(input=node_features, dim=1, index=edge_set.senders)
        receiver_features = torch.index_select(input=node_features, dim=1, index=edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        return self.edge_model(torch.cat(features, -1))
    
    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[1]
        features = [node_features]
        for edge_set in edge_sets:
            attention = self.attention_model(edge_set.features, edge_set.receivers)
            features.append(torch_scatter.scatter_add(torch.mul(edge_set.features, attention), edge_set.receivers, dim=1))
        return self.node_model(torch.cat(features, -1)), attention
    
    def forward(self, graph, residual=True, vis_att=False):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features, attention = self._update_node_features(graph.node_features, new_edge_sets)
        # node_features_residual = new_node_features.detach().clone()
        node_features_residual = attention.detach().clone()
        
        if residual:
            # add residual connections
            new_node_features += graph.node_features
            new_edge_sets = [es._replace(features=es.features + old_es.features)
                             for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        if not vis_att:
            return MultiGraph(new_node_features, new_edge_sets)
        else:
            return MultiGraph(new_node_features, new_edge_sets), node_features_residual


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""
    def __init__(self, make_mlp):
        super().__init__()
        self._make_mlp = make_mlp
        self.node_model = self._make_mlp([256, 128])
        self.edge_model = self._make_mlp([256, 128])

    def forward(self, graph, image_feature):
        self.image_feature = image_feature
        batch_size = self.image_feature.shape[0]
        node_num = graph.node_features.shape[0]
        node_image_feature = self.image_feature.view(batch_size, 1, self.image_feature.shape[1]).expand(-1, node_num, -1)

        # node_latents = self.node_model(graph.node_features)
        node_feature = graph.node_features
        # node_latents = node_latents.view(1, node_latents.shape[0], node_latents.shape[1]).expand(batch_size, -1, -1)
        node_feature = node_feature.view(1, node_feature.shape[0], node_feature.shape[1]).expand(batch_size, -1, -1)
        # node_latents = torch.cat([node_latents, node_image_feature], -1)
        tmp = node_feature.detach().clone()
        node_feature = torch.cat([node_feature, node_image_feature], -1)
        node_latents = self.node_model(node_feature)

        new_edges_sets = []
        for edge_set in graph.edge_sets:
            # latent = self._make_mlp(self._latent_size)(edge_set.features)
            # latent = self.edge_model(edge_set.features)
            edge_feature = edge_set.features
            # latent = latent.view(1, latent.shape[0], latent.shape[1]).expand(batch_size, -1, -1)
            edge_feature = edge_feature.view(1, edge_feature.shape[0], edge_feature.shape[1]).expand(batch_size, -1, -1)
            # edge_image_feature = self.image_feature.view(batch_size, 1, self.image_feature.shape[1]).expand(-1, edge_set.features.shape[0], -1)
            # edge_feature = torch.cat([edge_feature, edge_image_feature], -1)
            latent = self.edge_model(edge_feature)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets), tmp 


class Decoder(nn.Module):
    """Decodes node features from graph."""
    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        if type(graph) == torch.Tensor:
            # Now it is not a graph, but a node feature
            return self.model(graph)
        return self.model(graph.node_features)


class Processor(nn.Module):
    def __init__(self, make_mlp, output_size, message_passing_steps):
        super().__init__()
        self._submodules_ordered_dict = OrderedDict()
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphNetBlock(model_fn=make_mlp, output_size=output_size)
        self.submodules = nn.Sequential(self._submodules_ordered_dict)

    def forward(self, graph, read_intermediate, vis_att=False):
        if not read_intermediate:
            if not vis_att:
                return self.submodules(graph)
            else:
                residual_list = []
                for layer in self.submodules:
                    graph, residual = layer(graph, vis_att=vis_att)
                    residual_list.append(residual)
                return graph, residual_list
        else:
            result = [graph]
            for layer in self.submodules:
                graph = layer(graph)
                result.append(graph)
            return result 


class GAT(nn.Module):
    def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='GAT'):
        super(GAT, self).__init__()
        # self._latent_size = latent_size
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self.encoder = Encoder(make_mlp=self._make_mlp)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        if type(output_size) == int:
            widths = [self._latent_size] * self._num_layers + [output_size]
        elif type(output_size) == list:
            widths = output_size
        network = LazyMLP(widths) 
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph, image_feature, read_intermediate, vis_att=False):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph, tmp = self.encoder(graph, image_feature)
        if not read_intermediate:
            if not vis_att:
                latent_graph = self.processor(latent_graph, read_intermediate)
                # pdb.set_trace()
                return self.decoder(latent_graph) 
            else:
                latent_graph, node_feat_residual = self.processor(latent_graph, read_intermediate, vis_att)
                # return self.decoder(latent_graph), [self.decoder(x) for x in node_feat_residual]
                return self.decoder(latent_graph), node_feat_residual
        else:
            result = self.processor(latent_graph, read_intermediate)
            return [self.decoder(x) for x in result]

