import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pdb

import networkx as nx
from collections import defaultdict

from time import time

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, mul_bias=False, grpe_cross=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        self.mul_bias = mul_bias
        self.grpe_cross = grpe_cross

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, q, k, v, attn_bias=None, spatial_pos_query=None, spatial_pos_key=None):
        orig_q_size = q.size()
        batch_size = q.size(0)
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)

        # q, k, v: [n_graph, num_heads, n_node+1, att_size]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.grpe_cross:
            spatial_bias_query = torch.matmul(q.unsqueeze(3), spatial_pos_query.transpose(3, 4)).squeeze(3)
            spatial_bias_key = torch.matmul(k.unsqueeze(3), spatial_pos_key.transpose(3, 4)).squeeze(3)
            spatial_bias = spatial_bias_query + spatial_bias_key
            a = torch.matmul(q, k.transpose(2, 3)) + spatial_bias  # 不加edge_bais
            a = a * self.scale
        else:
            # Scaled Dot-Product Attention.
            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
            q = q * self.scale
            a = torch.matmul(q, k.transpose(2, 3))  # [n_graph, num_heads, n_node+1, n_node+1]
        if attn_bias is not None:
            if not self.mul_bias:
                a += attn_bias
            else:
                a *= attn_bias
        a = torch.softmax(a, dim=3)
        a = self.att_dropout(a)
        # a: [n_graph, num_heads, n_node+1, n_node+1]
        # v: [n_graph, num_heads, n_node+1, att_size]
        # x: [n_graph, num_heads, n_node+1, att_size]
        x = a.matmul(v)
        x = x.transpose(1, 2).contiguous()  # [n_graph, n_node+1, num_heads, att_size]
        x = x.view(batch_size, -1, self.num_heads * self.att_size)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, mul_bias=False, grpe_cross=False):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads, mul_bias, grpe_cross)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        self.self_attention.reset_parameters()

        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None, spatial_pos_query=None, spatial_pos_key=None):
        """
        x: [n_graph, n_node+1, hidden_size]，其中 n_graph 固定为 1，节点多一个是虚拟节点，hidden_size 需要和 init 部分对应
        """
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, spatial_pos_query, spatial_pos_key)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Graphormer(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        num_heads,
        hidden_dim,
        ffn_dim,
        grpe_cross=False,
        use_len_spd=True,
        use_num_spd=False,
        use_cnb_jac=False,
        use_cnb_aa=False,
        use_cnb_ra=False,
        use_degree=False,
        mul_bias=False,
        gravity_type=0,
        dropout_rate=0.1,
        intput_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        multi_hop_max_dist=20,
    ):
        super(Graphormer, self).__init__()
        self.grpe_cross = grpe_cross
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.hidden_dim = hidden_dim
        self.att_size = hidden_dim // num_heads
        self.use_len_spd = use_len_spd
        self.use_num_spd = use_num_spd
        self.use_cnb_jac = use_cnb_jac
        self.use_cnb_aa = use_cnb_aa
        self.use_cnb_ra = use_cnb_ra
        self.use_degree = use_degree
        self.gravity_type = gravity_type
        self.atom_encoder = nn.Linear(input_dim, hidden_dim)
        # self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
        # self.edge_type = edge_type
        # if self.edge_type == 'multi_hop':
        #    self.edge_dis_encoder = nn.Embedding(
        #        40 * num_heads * num_heads, 1)
        if self.grpe_cross:  # 按grpe的特征交叉方式，需要query和key两套结构特征embedding
            if use_len_spd:
                self.len_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.len_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.num_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_jac_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_aa_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_ra_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            # 固定0，不可学习
            self.padding1 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            self.padding2 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            # 随机初始化，可学习
            # self.padding1 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
            # self.padding2 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
        else:
            if use_len_spd:
                self.len_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder = nn.Embedding(40, num_heads, padding_idx=0)
        if use_degree:  # 点上的特征不需要两套
            self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.undir_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        num_edge_types = 1
        max_len_rule = 3
        num_rules = pow(num_edge_types*2, max_len_rule+1) - 2
        depth_rules = []
        for depth_rule in range(1, max_len_rule+1):
            depth_rules += [depth_rule] * pow(num_edge_types*2, depth_rule)
        self.depth_rules = torch.Tensor(depth_rules).long()
        self.gravity_scale = 1. / torch.pow(self.depth_rules, 2)
        if gravity_type in [1, 2, 3]:
            self.path_rule_lin = nn.Linear(num_rules, num_heads)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, mul_bias, grpe_cross)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.final_ln.reset_parameters()
        self.atom_encoder.reset_parameters()
        # self.edge_encoder.reset_parameters()
        # self.edge_type = edge_type
        # if self.edge_type == 'multi_hop':
        #    self.edge_dis_encoder.reset_parameters()
        if self.grpe_cross:
            if self.use_len_spd:
                self.len_shortest_path_encoder_query.reset_parameters()
                self.len_shortest_path_encoder_key.reset_parameters()
            if self.use_num_spd:
                self.num_shortest_path_encoder_query.reset_parameters()
                self.num_shortest_path_encoder_key.reset_parameters()
            if self.use_cnb_jac:
                self.undir_jac_encoder_query.reset_parameters()
                self.undir_jac_encoder_key.reset_parameters()
            if self.use_cnb_aa:
                self.undir_aa_encoder_query.reset_parameters()
                self.undir_aa_encoder_key.reset_parameters()
            if self.use_cnb_ra:
                self.undir_ra_encoder_query.reset_parameters()
                self.undir_ra_encoder_key.reset_parameters()
        else:
            if self.use_len_spd:
                self.len_shortest_path_encoder.reset_parameters()
            if self.use_num_spd:
                self.num_shortest_path_encoder.reset_parameters()
            if self.use_cnb_jac:
                self.undir_jac_encoder.reset_parameters()
            if self.use_cnb_aa:
                self.undir_aa_encoder.reset_parameters()
            if self.use_cnb_ra:
                self.undir_ra_encoder.reset_parameters()
        if self.gravity_type in [1, 2, 3]:
            self.path_rule_lin.reset_parameters()
        if self.use_degree:
            self.in_degree_encoder.reset_parameters()
            self.out_degree_encoder.reset_parameters()
            self.undir_degree_encoder.reset_parameters()

    # @TimerGuard('forward', 'utils')
    def forward(self, data, split_type='train_pos'):
        # attn_bias：图中节点对之间的最短路径距离超过最短路径限制最大距离(len_shortest_path_max)的位置为-∞，其余位置为0，形状为(n_graph, n_node+1, n_node+1)
        # len_shortest_path：图中节点对之间的最短路径长度，形状为(n_graph, n_node, n_node)
        # x：图中节点的特征，形状为(n_graph, n_node, n_node_features)
        # in_degree：图中节点的入度，形状为(n_graph, n_node)
        # out_degree：图中节点的出度，形状为(n_graph, n_node)
        # edge_input：图中节点对之间的最短路径(限制最短路径最大跳数为multi_hop_max_dist)上的边的特征，形状为(n_graph, n_node, n_node, multi_hop_max_dist, n_edge_features)
        # attn_edge_type：图的边特征，形状为(n_graph, n_node, n_node, n_edge_features)
        
        # 数据集中的形状为 (node_index, node_features)，Graphormer 需要的形状为 (graph, node_index, node_features)，在当前任务下只有一个图，因此使用 unsqueeze(0)
        # z_emb: [n_graph, n_head(2), dim_hidden]
        x = data['x'].unsqueeze(0)
        device = x.device

        # edge_input = data.edge_input
        # graph_attn_bias
        # 添加虚拟节点表示全图特征表示，之后按照图中正常节点处理
        n_graph, n_node = x.size()[:2]

        # attn_bias：图中节点对之间的最短路径距离超过最短路径限制最大距离(len_shortest_path_max)的位置为-∞，其余位置为0，形状为(n_graph, n_node+1, n_node+1)
        # 这里对路径距离暂时不作限制，默认为 0
        attn_bias = torch.zeros((n_graph, n_node+1, n_node+1))
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # 空间编码,节点之间最短路径长度对应的可学习标量
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = torch.zeros([n_graph, self.num_heads, n_node, n_node], device=device)
        spatial_pos_query = torch.zeros([n_graph, self.num_heads, n_node, n_node, self.att_size], device=device)
        spatial_pos_key = torch.zeros([n_graph, self.num_heads, n_node, n_node, self.att_size], device=device)

        # 动态获取当前split对应的指标字段名
        len_sp_name = "len_shortest_path"
        num_sp_name = "num_shortest_path"
        jac_name = "undir_jac"
        aa_name = "undir_aa"
        ra_name = "undir_ra"

        if self.grpe_cross:
            # [n_graph, n_node, n_node] -> [n_graph, n_head, n_node, n_node, att_size]
            if self.use_len_spd:
                print("name", split_type)
                print("data[len_sp_name]", data[len_sp_name].shape)
                len_shortest_path = torch.clamp(data[len_sp_name], min=0, max=39).long()
                spatial_pos_query = self.len_shortest_path_encoder_query(len_shortest_path).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key = self.len_shortest_path_encoder_key(len_shortest_path).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_num_spd:
                num_shortest_path = torch.clamp(data[num_sp_name], min=0, max=39).long()
                spatial_pos_query += self.num_shortest_path_encoder_query(num_shortest_path.long()).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.num_shortest_path_encoder_key(num_shortest_path.long()).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_jac:
                undir_jac_enc = torch.clamp(data[jac_name]*30, min=0, max=39).long()
                spatial_pos_query += self.undir_jac_encoder_query(undir_jac_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_jac_encoder_key(undir_jac_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_aa:
                undir_aa_enc = torch.clamp(data[aa_name]*10, min=0, max=39).long()
                spatial_pos_query += self.undir_aa_encoder_query(undir_aa_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_aa_encoder_key(undir_aa_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            if self.use_cnb_ra:
                undir_ra_enc = torch.clamp(data[ra_name]*10, min=0, max=39).long()
                spatial_pos_query += self.undir_ra_encoder_query(undir_ra_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
                spatial_pos_key += self.undir_ra_encoder_key(undir_ra_enc).reshape(n_graph, n_node, n_node, self.att_size, self.num_heads).permute(0, 4, 1, 2, 3)
            padding1_batch, padding2_batch = self.padding1.repeat(n_graph, 1, 1, n_node, 1), self.padding2.repeat(n_graph, 1, n_node+1, 1, 1)
            # [n_graph, n_head, n_node, n_node, att_size] -> [n_graph, n_head, n_node+1, n_node+1, att_size]
            spatial_pos_query = torch.cat((padding2_batch, torch.cat((padding1_batch, spatial_pos_query), dim=2)), dim=3)
            spatial_pos_key = torch.cat((padding2_batch, torch.cat((padding1_batch, spatial_pos_key), dim=2)), dim=3)
        else:
            # spatial pos：空间编码,节点之间最短路径长度对应的可学习标量
            # [n_graph, n_node, n_node] -> [n_graph, n_head, n_node, n_node]
            # import pdb; pdb.set_trace()
            if self.use_len_spd:
                len_shortest_path = torch.clamp(data[len_sp_name], min=0, max=39).long()
                spatial_pos_bias = self.len_shortest_path_encoder(len_shortest_path).permute(0, 3, 1, 2)
            if self.use_num_spd:
                num_shortest_path = torch.clamp(data[num_sp_name], min=0, max=39).long()
                spatial_pos_bias += self.num_shortest_path_encoder(num_shortest_path.long()).permute(0, 3, 1, 2)
            if self.use_cnb_jac:
                undir_jac_enc = torch.clamp(data[jac_name]*30, min=0, max=39).long()
                spatial_pos_bias += self.undir_jac_encoder(undir_jac_enc).permute(0, 3, 1, 2)
            if self.use_cnb_aa:
                undir_aa_enc = torch.clamp(data[aa_name]*10, min=0, max=39).long()
                spatial_pos_bias += self.undir_aa_encoder(undir_aa_enc).permute(0, 3, 1, 2)
            if self.use_cnb_ra:
                undir_ra_enc = torch.clamp(data[ra_name]*10, min=0, max=39).long()
                spatial_pos_bias += self.undir_ra_encoder(undir_ra_enc).permute(0, 3, 1, 2)

        if self.gravity_type == 1:
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_weight).permute(0, 3, 1, 2)
        elif self.gravity_type == 2:
            gravity_scale = self.gravity_scale.to(device)
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_weight * gravity_scale).permute(0, 3, 1, 2)
        elif self.gravity_type == 3:
            gravity_scale = self.gravity_scale.to(device)
            spatial_pos_bias = spatial_pos_bias + self.path_rule_lin(data.paths_log_weight * gravity_scale).permute(0, 3, 1, 2)
        if not self.grpe_cross:
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        # 所有节点都和虚拟节点直接有边相连，则所有节点和虚拟节点之间的最短路径长度为1
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 本来 attn_bias 形状为 (n_graph, n_node+1, n_node+1)，将其 unsqueeze(1) 后得到 (n_graph, n_head, n_node+1, n_node+1) 的 graph_attn_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        x = x.to(torch.float32)
        node_feature = self.atom_encoder(x)           # [n_graph, n_node, n_hidden]

        # 根据节点的入度、出度为每个节点分配两个实值嵌入向量，添加到节点特征中作为输入
        # 这里默认不用
        if self.use_degree:
            if hasattr(data, 'in_degree'):
                in_degree = torch.clamp(data.in_degree, min=0, max=63).long()
                out_degree = torch.clamp(data.out_degree, min=0, max=63).long()
                node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            else:
                undir_degree = torch.clamp(data.undir_degree, min=0, max=63).long()
                node_feature = node_feature + self.undir_degree_encoder(undir_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            if self.grpe_cross:
                output = enc_layer(output, graph_attn_bias, spatial_pos_query, spatial_pos_key)
            else:
                output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        return output



class SubgraphGraphormer(Graphormer):
    def __init__(
        self,
        n_layers,
        input_dim,
        num_heads,
        hidden_dim,
        ffn_dim,
        device,
        grpe_cross=False,
        use_len_spd=True,
        use_num_spd=False,
        use_cnb_jac=False,
        use_cnb_aa=False,
        use_cnb_ra=False,
        use_degree=False,
        mul_bias=False,
        gravity_type=0,
        dropout_rate=0.1,
        intput_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        multi_hop_max_dist=20,
    ):
        super(Graphormer, self).__init__()
        self.grpe_cross = grpe_cross
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.hidden_dim = hidden_dim
        self.att_size = hidden_dim // num_heads
        self.use_len_spd = use_len_spd
        self.use_num_spd = use_num_spd
        self.use_cnb_jac = use_cnb_jac
        self.use_cnb_aa = use_cnb_aa
        self.use_cnb_ra = use_cnb_ra
        self.use_degree = use_degree
        self.gravity_type = gravity_type
        self.atom_encoder = nn.Linear(input_dim, hidden_dim)
        # self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
        # self.edge_type = edge_type
        # if self.edge_type == 'multi_hop':
        #    self.edge_dis_encoder = nn.Embedding(
        #        40 * num_heads * num_heads, 1)
        if self.grpe_cross:  # 按grpe的特征交叉方式，需要query和key两套结构特征embedding
            if use_len_spd:
                self.len_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.len_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.num_shortest_path_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_jac_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_aa_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder_query = nn.Embedding(40, hidden_dim, padding_idx=0)
                self.undir_ra_encoder_key = nn.Embedding(40, hidden_dim, padding_idx=0)
            # 固定0，不可学习
            self.padding1 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            self.padding2 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.att_size), requires_grad=False)
            # 随机初始化，可学习
            # self.padding1 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
            # self.padding2 = nn.Parameter(torch.randn(1, num_heads, 1, 1, self.att_size))
        else:
            if use_len_spd:
                self.len_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_num_spd:
                self.num_shortest_path_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_jac:
                self.undir_jac_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_aa:
                self.undir_aa_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            if use_cnb_ra:
                self.undir_ra_encoder = nn.Embedding(40, num_heads, padding_idx=0)
        if use_degree:  # 点上的特征不需要两套
            self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.undir_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)

        num_edge_types = 1
        max_len_rule = 3
        num_rules = pow(num_edge_types*2, max_len_rule+1) - 2
        depth_rules = []
        for depth_rule in range(1, max_len_rule+1):
            depth_rules += [depth_rule] * pow(num_edge_types*2, depth_rule)
        self.depth_rules = torch.Tensor(depth_rules).long()
        self.gravity_scale = 1. / torch.pow(self.depth_rules, 2)
        if gravity_type in [1, 2, 3]:
            self.path_rule_lin = nn.Linear(num_rules, num_heads)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, mul_bias, grpe_cross)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.device = device

    def _parse_all_mask(self, all_mask, batch_u, batch_v, num_nodes):
        """解析all_mask并强制包含u,v节点到对应子图"""
        subgraphs = defaultdict(list)
        # 原始解析逻辑
        if isinstance(all_mask, np.ndarray):
            all_mask = torch.from_numpy(all_mask)
        for col in range(all_mask.size(1)):
            graph_idx = all_mask[0, col].item()
            node_idx = all_mask[1, col].item()
            subgraphs[graph_idx].append(node_idx)
        
        # 强制将u,v加入对应子图
        for i, (u, v) in enumerate(zip(batch_u, batch_v)):
            subgraphs[i].extend([u.item(), v.item()])  # 确保当前批次的u,v加入子图i
        
        # 去重并排序
        subgraph_list = []
        for idx in range(len(subgraphs)):
            nodes = torch.unique(torch.tensor(subgraphs[idx], dtype=torch.long))
            subgraph_list.append(nodes)
        
        # 创建u,v位置映射表
        u_positions = []
        v_positions = []
        for i, sub_nodes in enumerate(subgraph_list):
            node_list = sub_nodes.tolist()
            u_global = batch_u[i].item()
            v_global = batch_v[i].item()
            u_pos = node_list.index(u_global) + 1  # +1因为包含graph_token
            v_pos = node_list.index(v_global) + 1
            u_positions.append(u_pos)
            v_positions.append(v_pos)
        
        return subgraph_list, (u_positions, v_positions)

    def _build_subgraph_features(self, data, sub_nodes, device):
        """给定一个子图，构建子图中的特征矩阵"""
        k = len(sub_nodes)
        # 节点特征编码
        # 使用传播后的特征
        sub_x = data['re_feature_x'][sub_nodes].unsqueeze(0)  # [1, k, feat_dim]
        node_feature = self.atom_encoder(sub_x)
        
        # 添加虚拟节点
        graph_token = self.graph_token.weight.unsqueeze(0)  # [1, 1, hidden_dim]
        return torch.cat([graph_token, node_feature], dim=1)  # [1, k+1, hidden_dim]

    def forward(self, data, all_mask, batch_u, batch_v):
        # 解析掩码矩阵
        subgraphs, (u_pos, v_pos) = self._parse_all_mask(all_mask, batch_u, batch_v, data["num_nodes"])
        batch_size = len(subgraphs)
        device = data['x'].device
        # 批量特征准备
        batch_features = []
        batch_attn_bias = []
        spatial_pos_queries = []
        spatial_pos_keys = []

        # 先按照子图遍历
        for sub_nodes in subgraphs:
            k = len(sub_nodes)
            if k == 0:
                continue  # 跳过空子图

            # 构建子图特征
            graph_features = self._build_subgraph_features(data, sub_nodes, device)
            batch_features.append(graph_features)

            # 计算子图指标
            metrics = self._compute_subgraph_metrics(data, sub_nodes.tolist())
            
            # 构建注意力偏置
            attn_bias = torch.zeros((1, self.num_heads, k+1, k+1), device=device)
            if not self.grpe_cross:
                spatial_bias = self._get_spatial_bias(metrics, k)
                attn_bias[:, :, 1:, 1:] = spatial_bias

            # 处理虚拟节点连接
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            attn_bias[:, :, 1:, 0] += t
            attn_bias[:, :, 0, :] += t
            batch_attn_bias.append(attn_bias)

            # 处理GRPE交叉特征
            if self.grpe_cross:
                query, key = self._build_grpe_features(metrics, k)
                spatial_pos_queries.append(query)
                spatial_pos_keys.append(key)
        # 批量对齐处理
        
        max_k = max(f.size(1) for f in batch_features)

        padded_features = []
        padded_attn_bias = []
        for feat, bias in zip(batch_features, batch_attn_bias):
            pad_size = max_k - feat.size(1)
            padded_features.append(F.pad(feat, (0,0,0,pad_size)))
            padded_attn_bias.append(F.pad(bias, (0,pad_size,0,pad_size)))
            
        batch_x = torch.cat(padded_features, dim=0)
        batch_attn = torch.cat(padded_attn_bias, dim=0)
        # Transformer编码
        output = self.input_dropout(batch_x)
        for enc_layer in self.layers:
            if self.grpe_cross:
                spatial_q = self._pad_grpe_features(spatial_pos_queries, max_k)
                spatial_k = self._pad_grpe_features(spatial_pos_keys, max_k)
                output = enc_layer(
                    output, 
                    batch_attn, 
                    spatial_q, 
                    spatial_k
                )
            else:
                output = enc_layer(output, batch_attn)

        return self.final_ln(output), (u_pos, v_pos)

    def _compute_subgraph_metrics(self, data, sub_nodes):
        """优化的子图指标计算"""
        edge_index = data["edge_index"].cpu().numpy()
        k = len(sub_nodes)
        node_map = {n: i for i, n in enumerate(sub_nodes)}
        
        # 构建NetworkX子图
        G = nx.Graph()
        G.add_nodes_from(range(k))
        for u, v in edge_index.T:
            if u in node_map and v in node_map:
                G.add_edge(node_map[u], node_map[v])
                
        # 预计算所有节点对的最短路径
        path_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.multi_hop_max_dist))
        
        # 初始化指标矩阵
        metrics = {
            'len_shortest_path': np.full((k, k), self.multi_hop_max_dist),
            'undir_jac': np.zeros((k, k)),
            'undir_aa': np.zeros((k, k)),
            'undir_ra': np.zeros((k, k))
        }

        # 并行计算指标
        for i in range(k):
            for j in range(i+1, k):
                # 最短路径
                metrics['len_shortest_path'][i,j] = path_lengths[i].get(j, self.multi_hop_max_dist)
                
                # 共同邻居指标
                cn = sorted(nx.common_neighbors(G, i, j))
                if cn:
                    deg = np.array([G.degree(n) for n in cn])
                    metrics['undir_jac'][i,j] = len(cn)/(G.degree(i)+G.degree(j)-len(cn))
                    metrics['undir_aa'][i,j] = np.sum(1/np.log(deg + 1e-5))
                    metrics['undir_ra'][i,j] = np.sum(1/deg)
                    
                metrics['len_shortest_path'][j,i] = metrics['len_shortest_path'][i,j]
                metrics['undir_jac'][j,i] = metrics['undir_jac'][i,j]
                metrics['undir_aa'][j,i] = metrics['undir_aa'][i,j]
                metrics['undir_ra'][j,i] = metrics['undir_ra'][i,j]
                
        # 转换为tensor并截断
        return {k: torch.clamp(torch.tensor(v), 0, 39).long().to(data["x"].device) 
               for k, v in metrics.items()}

    def _build_grpe_features(self, metrics, k):
        """构建GRPE交叉模式的结构特征嵌入"""
        # 初始化query和key的特征矩阵
        spatial_query = torch.zeros(1, k, k, self.hidden_dim, 
                                device=metrics['len_shortest_path'].device)
        spatial_key = torch.zeros_like(spatial_query)
        
        # 遍历所有启用的结构特征进行累加
        if self.use_len_spd:
            len_spd = metrics['len_shortest_path']  # [k, k]
            # Query侧嵌入
            spatial_query += self.len_shortest_path_encoder_query(len_spd).unsqueeze(0)  # [1,k,k,hid]
            # Key侧嵌入
            spatial_key += self.len_shortest_path_encoder_key(len_spd).unsqueeze(0)
            
        if self.use_num_spd and False:
            num_spd = metrics['num_shortest_path']  # 需确保metrics中存在该字段
            spatial_query += self.num_shortest_path_encoder_query(num_spd).unsqueeze(0)
            spatial_key += self.num_shortest_path_encoder_key(num_spd).unsqueeze(0)
            
        if self.use_cnb_jac:
            jac = metrics['undir_jac']
            spatial_query += self.undir_jac_encoder_query(jac).unsqueeze(0)
            spatial_key += self.undir_jac_encoder_key(jac).unsqueeze(0)
            
        if self.use_cnb_aa:
            aa = metrics['undir_aa']
            spatial_query += self.undir_aa_encoder_query(aa).unsqueeze(0)
            spatial_key += self.undir_aa_encoder_key(aa).unsqueeze(0)
            
        if self.use_cnb_ra:
            ra = metrics['undir_ra']
            spatial_query += self.undir_ra_encoder_query(ra).unsqueeze(0)
            spatial_key += self.undir_ra_encoder_key(ra).unsqueeze(0)
        
        # 调整维度为多头注意力格式 [1, num_heads, k, k, att_size]
        spatial_query = spatial_query.view(1, k, k, self.num_heads, self.att_size)
        spatial_query = spatial_query.permute(0, 3, 1, 2, 4)  # [1, h, k, k, d]
        
        spatial_key = spatial_key.view(1, k, k, self.num_heads, self.att_size)
        spatial_key = spatial_key.permute(0, 3, 1, 2, 4)
        
        # 添加padding参数（根据__init__中的定义）
        spatial_query += self.padding1[:, :, :k, :k, :]
        spatial_key += self.padding2[:, :, :k, :k, :]
        
        return spatial_query, spatial_key
    
    def _pad_grpe_features(self, features, max_k):
        """将不同子图的GRPE特征填充到最大尺寸"""
        if not features:  # 处理空列表情况
            return torch.zeros(
                1, self.num_heads, max_k, max_k, self.att_size,
                device=self.padding1.device  # 保持设备一致
            )
        
        padded = []
        for feat in features:
            # 原始特征形状: [1, num_heads, k, k, att_size]
            b, h, k, _, d = feat.shape
            pad_size = max_k - k
            
            # 在第三、四维度进行右侧填充
            padded_feat = F.pad(
                feat, 
                (0, 0, 0, pad_size, 0, pad_size, 0, 0, 0, 0), 
                value=0
            )  # 输出形状: [1, h, max_k, max_k, d]
            padded.append(padded_feat)
        
        return torch.cat(padded, dim=0)  # 合并batch维度 [bs, h, max_k, max_k, d]