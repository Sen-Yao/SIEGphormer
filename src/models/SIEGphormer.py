import math
import torch
import torch.nn as nn
from torch_scatter import scatter 
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch.nn.init import uniform_
import numpy as np 

import time
import sys

from models.other_models import *
from models.graphormer import Graphormer, SubgraphGraphormer
from modules.node_encoder import NodeEncoder
from modules.layers import LinkTransformerLayer, GraphTransformer

from util.utils import torch_sparse_tensor_to_sparse_mx, sparse_mx_to_torch_sparse_tensor, drnl_node_labeling, drnl_subgraph_labeling


import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")



class SIEGphormer(nn.Module):
    """
    Transformer for Link Prediction
    """
    def __init__(
        self,
        train_args,
        data,
        global_logger,
        device="cuda"
    ):
        super().__init__()
        
        self.train_args = train_args
        self.data = data
        self.global_logger = global_logger
        self.device = device

        # PPR Thresholds
        self.thresh_cn = train_args['thresh_cn']
        self.thresh_1hop = train_args['thresh_1hop']
        self.thresh_non1hop = train_args['thresh_non1hop']

        # Allows for easier way to know if we need to ignore some set of nodes
        # Also allows us to be more efficient
        # 根据超参数，选择此时模型的「mask」
        if self.thresh_non1hop == 1 and self.thresh_1hop == 1:
            self.mask = "cn"
        elif self.thresh_non1hop == 1 and self.thresh_1hop < 1:
            self.mask = "1-hop"
        else:
            self.mask = "all"
            
        self.dim = train_args['dim']
        self.att_drop = train_args.get('att_drop', 0)
        self.num_layers = train_args['trans_layers']
        self.num_nodes = data['x'].shape[0]
        self.out_dim = self.dim * 6

        self.gnn_norm = nn.LayerNorm(self.dim)
        self.node_encoder = NodeEncoder(data, train_args, device=device)

        self.att_layers = nn.ModuleList()
        att_inner_dim = self.dim * 2 if self.num_layers > 1 else self.dim

        self.att_layers.append(LinkTransformerLayer(self.dim, train_args, out_dim=att_inner_dim))
        for _ in range(self.num_layers-2):
            self.att_layers.append(LinkTransformerLayer(self.dim, train_args, node_dim=self.dim))
        if self.num_layers > 1:
            self.att_layers.append(LinkTransformerLayer(self.dim, train_args, out_dim=self.dim, node_dim=self.dim))

        self.elementwise_lin = MLP(2, self.dim, self.dim, self.dim)
        
        # Structural info
        self.ppr_encoder_cn = MLP(2, 2, self.dim, self.dim)
        if self.mask == "cn":
            count_dim = 1
        elif self.mask == "1-hop":
            self.ppr_encoder_onehop = MLP(2, 2, self.dim, self.dim)
            count_dim = 3
        else:
            count_dim = 4
            self.ppr_encoder_onehop = MLP(2, 2, self.dim, self.dim)
            self.ppr_encoder_non1hop = MLP(2, 2, self.dim, self.dim)

        
        pairwise_dim = self.dim * train_args['num_heads'] + count_dim
        self.pairwise_lin = MLP(2, pairwise_dim, pairwise_dim, self.dim)          

        # 矩阵传播用
        self.K = self.train_args['mat_prop']  # 从参数中获取 K 值
        self.alpha = self.train_args['alpha']
        self.alpha = nn.Parameter(torch.zeros(self.K + 1))

        # 把输入矩阵做 MLP 再传播
        self.feature_proj = nn.Sequential(
            nn.Linear(data['x'].shape[1], self.dim)
            # nn.LayerNorm(self.dim),
            # nn.ReLU()
        )

        self.drnl = torch.zeros((self.train_args['batch_size'], self.data['num_nodes']), device=device)
        # print("init_drnl", self.drnl.dtype)

        self.to(device=device)

        # 对各个启发式方法应用 MLP 进行编码
        self.CN_encoder = MLP(2, 2, self.dim, self.dim)
        self.AA_encoder = MLP(2, 2, self.dim, self.dim)
        self.PPR_encoder = MLP(2, 2, self.dim, self.dim)
        self.DRNL_encoder = MLP(2, 2, self.dim, self.dim)

        self.structure_encoder = MLP(2, 4 * self.dim, self.dim, self.dim)

        self.transformer = GraphTransformer(
            dim= 2 * self.dim,
            num_heads=train_args['num_heads'],
            dropout=self.att_drop
        )

    def forward(self, batch, adj_prop=None, adj_mask=None, test_set=False, return_weights=False):
        """
        Calculate edge representations

        Parameters:
        ----------
            batch: torch.Tensor
                2 x BS Tensor that hold source and target nodes
            test_set: bool
                Whether evaluating on test set. Needed if using val_edges in agg

        Returns:
        --------
        torch.Tensor
            BS x self.dim
        """
        # 林子垚：
        # batch 是当前 batch 传入的那些需要进行预测的边，形状为（2, batch_size), 表示了每个样本的目标节点对下标
        # adj_prop 仅在 mask_input 时生效，只有 Pubmed 数据集有此设置，否则为 None
        # adj_mask 是掩码的稀疏邻接矩阵，用于记录忽略了样本后，有哪些节点是相连的
        # test_set 表示当前为测试集
        # return_weights 为 debug 用


        batch = batch.to(self.device)

        # X_node = self.propagate(adj_prop, test_set)
        # 原本的节点特征通过上面的传播得到，现在直接读数据
        
        X_node = self.data['x']
        # X_node 是节点特征矩阵，形状为 (node_index, node_features)
        X_node = self.re_features(X_node, self.K)
        self.data['re_feature_x'] = X_node
        
        # 全局计算 DRNL
        
        # batch[0] 为长度为 batch_size 的一个一维向量，表示了这个 batch 所有的目标节点对的 a 节点，同理 batch[1] 表示了 b 节点
        
        subg_mask = self.khop_sampling(batch=batch, k=2)
        heuristic_index = self.cal_heuristic(batch, subg_mask)
        h_tokens = self.tokenizer(subg_mask, heuristic_index)
        
        # h_tokens 形状为 (num_samples, num_nodes, 2 * self.dim)，表示了每个节点的结构信息和节点特征拼接后的结果
        h_a, h_b, h_CLS = self.transformer(h_tokens, batch, subg_mask)
        # print(f"After transformer", file=sys.stderr)
        # 三者形状为 (num_samples,  2 * self.dim)，表示了每个样本中的 a 节点、b 节点和 CLS 节点的表示

        combined_feats = torch.cat((h_a, h_b, h_CLS), dim=-1)
        # combined_feats 形状为 (num_samples, 6 * self.dim)，表示了每个样本中 a 节点、b 节点和 CLS 节点的表示拼接后的结果

        return combined_feats
    

    def propagate(self, adj=None, test_set=False):
        """
        Propagate via GNN

        Returns:  
        -------
        torch.Tensor
            |V| x self.dim
        """
        if adj is None:
            adj = self.get_adj(test_set)
        x = self.data['x']

        if "emb" in self.data:
            x = self.data['emb'](x)

        X_node = self.node_encoder(x, adj, test_set)        
        X_node = self.gnn_norm(X_node)

        return X_node

        """
        Get PPR scores for non-1hop nodes.

        NOTE: Use original adj_mask (in train_model.train_epoch we remove the batch links)
        Done since removing them converts src/tgt to >1-hop nodes
        Therefore removing CN and 1-hop will also remove the batch links
        Don't bother in testing since we are only given the train graph
        """
        adj = self.get_adj(test_set, mask=True)
        src_adj = torch.index_select(adj, 0, batch[0])
        tgt_adj = torch.index_select(adj, 0, batch[1])

        ppr = self.get_ppr(test_set)
        src_ppr = torch.index_select(ppr, 0, batch[0])
        tgt_ppr = torch.index_select(ppr, 0, batch[1])

        # Remove CN scores
        src_ppr = src_ppr - src_ppr * (src_adj * tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj * tgt_adj)
        # Also need to remove CN entries in Adj, otherwise leak into next computation
        src_adj = src_adj - src_adj * (src_adj * tgt_adj)
        tgt_adj = tgt_adj - tgt_adj * (src_adj * tgt_adj)

        # Remove 1-Hop scores
        src_ppr = src_ppr - src_ppr * (src_adj + tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj + tgt_adj)

        # Want to make sure we include both when we convert to dense so indices align
        # Do so by adding 1 to each based on the other
        src_ppr_add = src_ppr + torch.sign(tgt_ppr)
        tgt_ppr_add = tgt_ppr + torch.sign(src_ppr)

        src_ix = src_ppr_add.coalesce().indices()
        src_vals = src_ppr_add.coalesce().values()
        # tgt_ix = tgt_ppr_add.coalesce().indices()
        tgt_vals = tgt_ppr_add.coalesce().values()

        # Now we can remove value which is just 1
        # NOTE: This technically creates -1 scores for ppr scores that were 0 for src and tgt
        # Doesn't matter as they'll be filtered out by condition later 
        src_vals = src_vals - 1
        tgt_vals = tgt_vals - 1

        ppr_condition = (src_vals >= self.thresh_non1hop) & (tgt_vals >= self.thresh_non1hop)
        src_ix, src_vals, tgt_vals = src_ix[:, ppr_condition], src_vals[ppr_condition], tgt_vals[ppr_condition]

        return src_ix, src_vals, tgt_vals

    def re_features(self, features, K):
        # 林子垚：
        
        # features 是节点特征矩阵
        # K 是传播的过程执行 K 次

        features = self.feature_proj(features)

        # adj：是已经归一化的 torch.sparse 邻接矩阵
        adj = self.data['norm_adj']

        # nodes_features 是一个空张量，用于存储每次邻接矩阵传播的结果
        nodes_features = [features]
        x = features.clone()
        for _ in range(K):
            x = torch.sparse.mm(adj, x)
            nodes_features.append(x)
        # nodes_features = torch.stack(nodes_features, dim=1).unsqueeze(1)
        nodes_features = torch.stack(nodes_features, dim=1)
        # print(nodes_features.shape)

        # 如果固定权重，用这套：
        weighted_features = nodes_features.unsqueeze(1)[:, 0, -1, :] * (1-self.train_args['alpha']) + features * self.train_args['alpha']

        # 如果动态 alpha，用这套：
        # weights = torch.softmax(self.alpha, dim=0)
        # weighted_features = (nodes_features * weights.view(1, -1, 1)).sum(dim=1)

        return weighted_features
    
    def khop_sampling(self, batch, k=3):
        """
        根据全局的邻接矩阵和 batch 中的节点对，进行 k-hop 邻接矩阵采样。若某节点存在于两个目标节点对之一的 k 跳内，则该节点会被采样到。
        每个样本都有对应的节点对，从而采样出对应的一个子图。
        
        此函数输出结果为压缩后的二维 COO 坐标形式的稠密子图掩码矩阵 subg_mask，格式为 (sample_idx, node_idx)
        其中 sample_idx 表示了当前 batch 中的样本的下标，node_idx 表示了子图中每个节点在全图的下标。
        最终矩阵的每个元素都代表了 sample_idx 对应的样本中，对应的节点存在于子图中。
        """
        # 邻接矩阵为 data['adj_mask']，这是一个 COO 格式的稀疏矩阵，表示了全图的邻接矩阵
        # batch[0] 是一个目标节点的列表，batch[1] 是另一个目标节点的列表
        adj = self.data['adj_mask']
        src_list, dst_list = batch[0].tolist(), batch[1].tolist()

        # 初始化结果列表
        subg_masks = []

        for sample_idx, (src, dst) in enumerate(zip(src_list, dst_list)):
            # 初始化当前样本的子图节点集合
            src_nodes = {src}
            dst_nodes = {dst}

            # BFS 进行 k-hop 采样
            for _ in range(k):
                # 获取当前 hop 的邻居
                src_neighbors = set(adj.indices()[1][torch.isin(adj.indices()[0], torch.tensor(list(src_nodes), device=adj.device))].tolist())
                dst_neighbors = set(adj.indices()[1][torch.isin(adj.indices()[0], torch.tensor(list(dst_nodes), device=adj.device))].tolist())

                # 更新节点集合，避免重复采样
                src_nodes.update(src_neighbors)
                dst_nodes.update(dst_neighbors)

            # 合并两个目标节点的 k-hop 邻居集合
            k_hop_nodes = src_nodes.union(dst_nodes)

            # 将子图节点集合转换为 COO 格式
            for node in k_hop_nodes:
                subg_masks.append((sample_idx, node))

        # 转换为张量形式
        subg_mask = torch.tensor(subg_masks, dtype=torch.long).t()

        return subg_mask
    
    def cal_heuristic(self, batch, subg_mask):
        """
        根据输入子图掩码，计算相关的结构性指标，将其存储。用于后续进行 Tokenize

        输入：
        batch: 形状为（2, batch_size）的张量。每一行分别表示源节点和目标节点的下标。
        subg_mask: torch.Tensor, 压缩后的二维 COO 坐标形式的稠密子图掩码矩阵，格式为 (2, n)， n 为当前 batch 中所有处于子图中的节点个数。

        输出：
        heuristic_index: 对所有子图中节点计算出的启发式指标
        """

        # 初始化启发式指标矩阵，形状为 (n, 4, 2)
        n = subg_mask.size(1)
        heuristic_index = torch.zeros((n, 4, 2), device=self.device)

        # 获取目标节点对 (a, b)
        a_indices = batch[0]
        b_indices = batch[1]

        ppr = self.data["ppr"].to_dense()


        # 获取子图中的节点索引
        sample_indices = subg_mask[0]
        node_indices = subg_mask[1]


        # 获取 PPR(a, u) 和 PPR(b, u)
        ppr_a = torch.index_select(ppr, 0, a_indices)[sample_indices, node_indices]
        ppr_b = torch.index_select(ppr, 0, b_indices)[sample_indices, node_indices]
        heuristic_index[:, 2, 0] = ppr_a
        heuristic_index[:, 2, 1] = ppr_b

        # 获取 DRNL(u)
        self.drnl = drnl_node_labeling(self.data['coo_adj'].tocsr(), batch[0], batch[1])
        drnl_values = self.drnl[sample_indices, node_indices]
        heuristic_index[:, 3, 0] = drnl_values
        heuristic_index[:, 3, 1] = drnl_values

        return heuristic_index




    def tokenizer(self, subg_mask, heuristic_index):
        """
        使用可学习层对其进行编码后，将子图中的每个节点 tokenize 为一个向量，提供给后续的 Transformer 模块进行处理。

        输入：
        subg_mask: torch.Tensor, 压缩后的二维 COO 坐标形式的稠密子图掩码矩阵，格式为 (sample_idx, node_idx)，形状为 (2, n)， n 为当前 batch 中所有处于子图中的节点个数。
        其中 sample_idx 表示了当前 batch 中的样本的下标，node_idx 表示了子图中每个节点在全图的下标。
        最终矩阵的每个元素都代表了 sample_idx 对应的样本中，对应的节点存在于子图中。
        heuristic_index: torch.Tensor, 启发式指标矩阵，形状为 (n, 4, 2)， n 为当前 batch 中所有处于子图中的节点个数。与 subg_mask 一一对应。

        输出：
        h_tokens: torch.Tensor, 经过可学习层编码后的子图节点向量，形状为 (num_samples, num_nodes, 2 * self.dim)
        """
        # 编码 CN(a, u) 和 CN(b, u)
        cn_encoded = self.CN_encoder(heuristic_index[:, 0, :])

        # 编码 AA(a, u) 和 AA(b, u)
        aa_encoded = self.AA_encoder(heuristic_index[:, 1, :])

        # 编码 PPR(a, u) 和 PPR(b, u)
        ppr_encoded = self.PPR_encoder(heuristic_index[:, 2, :])

        # 编码 DRNL(u)
        drnl_encoded = self.DRNL_encoder(heuristic_index[:, 3, :])

        # 拼接四种结构信息
        structure_info = torch.cat((cn_encoded, aa_encoded, ppr_encoded, drnl_encoded), dim=-1)

        # 映射到 self.dim 的张量
        structure_info = self.structure_encoder(structure_info)

        # 获取全局节点特征
        node_features = self.data['re_feature_x'][subg_mask[1]]

        # 拼接结构信息和节点特征
        h_tokens = torch.cat((structure_info, node_features), dim=-1)

        return h_tokens
    