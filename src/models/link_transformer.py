import math
import torch
import torch.nn as nn
from torch_scatter import scatter 
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch.nn.init import uniform_
import numpy as np 

import time

from models.other_models import *
from models.graphormer import *
from modules.node_encoder import NodeEncoder
from modules.layers import LinkTransformerLayer

from util.utils import torch_sparse_tensor_to_sparse_mx, sparse_mx_to_torch_sparse_tensor, drnl_node_labeling, drnl_subgraph_labeling


import warnings
warnings.filterwarnings("ignore")



class LinkTransformer(nn.Module):
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
        if train_args['graphormer']:
            self.out_dim = self.dim * 2
        else:
            self.out_dim = self.dim * 2

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

        self.drnl_encoder = MLP(2, 1, self.dim, self.dim)
        
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

        self.graphormer = SubgraphGraphormer(n_layers=3,
                                     input_dim=self.dim,
                                     num_heads=1,
                                     hidden_dim=self.dim,
                                     ffn_dim=self.dim,
                                     grpe_cross=True,
                                     use_len_spd=True,
                                     use_num_spd=True,
                                     use_cnb_jac=True,
                                     use_cnb_aa=True,
                                     use_cnb_ra=True,
                                     use_degree=False,
                                     mul_bias=False,
                                     gravity_type=0,
                                     device=self.device)
        # self.graphormer.to(device=device)
        self.graphormer_encoder =  PairMLP(
            num_layers=3, 
            in_channels=128,  # u_emb和v_emb的输入维度
            hid_channels=512, 
            out_channels=self.dim, 
            norm="layer",
            merge_method="concat"
        )
        self.to(device=device)

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

        # print("X_node=", X_node.shape)

        # 计算子图掩码
        # self.subgraph_mask = self.cal_subgraph_mask(batch, test_set, adj_mask)
        
        # 全局计算 DRNL
        if True:
            self.drnl = drnl_node_labeling(self.data['coo_adj'].tocsr(), batch[0], batch[1])
            # print("forward", self.drnl.dtype)
            # drnl 形状为 (batch_size, num_nodes) 的稀疏矩阵

            

        # batch[0] 为长度为 batch_size 的一个一维向量，表示了这个 batch 所有的目标节点对的 a 节点，同理 batch[1] 表示了 b 节点
        x_i, x_j = X_node[batch[0]], X_node[batch[1]]
        elementwise_edge_feats = self.elementwise_lin(x_i * x_j) # 论文注释: h_a * h_b，但是这里好像加了个线性层？

        # att_weights 不重要，作者用来 debug 的        

        # 计算节点对的那些相关信息
        pairwise_feats, att_weights = self.calc_pairwise(batch, X_node, test_set, adj_mask=adj_mask, return_weights=return_weights)

        # Graphormer 部分
        if self.train_args['graphormer']:
            cn_info, onehop_info, non1hop_info = self.compute_node_mask(batch, test_set, adj_mask)
            subg_mask = torch.cat((cn_info[0], onehop_info[0], non1hop_info[0]), dim=-1)
            h_graphormer, (u_pos, v_pos) = self.graphormer(self.data, subg_mask, batch[0], batch[1])  # 输出应为 (batchsize, max_k+3, sefl.dim)
            batch_size = h_graphormer.size(0)
            u_emb = h_graphormer[torch.arange(batch_size), u_pos, :]
            v_emb = h_graphormer[torch.arange(batch_size), v_pos, :]
            graphormer_output = self.graphormer_encoder(u_emb, v_emb)
            combined_feats = torch.cat((elementwise_edge_feats, pairwise_feats, graphormer_output), dim=-1)
        else:
            combined_feats = torch.cat((elementwise_edge_feats, pairwise_feats), dim=-1)

        return combined_feats if not return_weights else (combined_feats, att_weights)
    

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


    def calc_pairwise(self, batch, X_node, test_set=False, adj_mask=None, return_weights=False):
        """
        Calculate the pairwise features for the node pairs


        Returns:
        -------
        torch.Tensor
            BS x self.dim
        """
        # 计算节点对的那些相关信息
        # 提取目标节点对的 feat
        k_i, k_j = X_node[batch[0]], X_node[batch[1]]
        # 拼接在一起
        pairwise_feats = torch.cat((k_i, k_j), dim=-1)


        if self.mask == "cn":
            cn_info, _, _ = self.compute_node_mask(batch, test_set, adj_mask)
            node_mask = cn_info[0]
            pes = self.get_pos_encodings(cn_info)

            for l in range(self.num_layers):
                pairwise_feats, att_weights = self.att_layers[l](node_mask, pairwise_feats, X_node, pes, return_weights=return_weights)
            
            num_cns = self.get_count(node_mask, batch, test_set)
            pairwise_feats = torch.cat((pairwise_feats, num_cns), dim=-1)

        else:
            # 算出三个级别的子图的相关信息，相互应该不包括
            cn_info, onehop_info, non1hop_info = self.compute_node_mask(batch, test_set, adj_mask)
            
            if non1hop_info is not None:
                all_mask = torch.cat((cn_info[0], onehop_info[0], non1hop_info[0]), dim=-1)
                
                # 传统 DRNL 或子图 DRNL
                # 传统 DRNL：此外，还需要在 forward 时启动 self.drnl 的更新
                subg_drnl = self.drnl[all_mask[0, :], all_mask[1, :]]

                # 子图 DRNL
                # subg_drnl = drnl_subgraph_labeling(self.data['coo_adj'].tocsr(), batch[0], batch[1], all_mask)
                
                drnl_info = (all_mask, subg_drnl)

                pes = self.get_pos_encodings(cn_info, onehop_info, non1hop_info, drnl_info)
            else:
                all_mask = torch.cat((cn_info[0], onehop_info[0]), dim=-1)
                pes = self.get_pos_encodings(cn_info, onehop_info)
            # all_mask 是稀疏矩阵形式的子图掩码。包括三个或两个级别的所有节点。形状为 [2, 整个 batch 中被采样到子图中的节点个数]
            # pairwise_feats 是一个稠密矩阵，形状为 [batchsize, 2 * self.dim]，表示了每个节点对的特征被拼接在一起
            # pes 是所谓的位置编码，如果开启了 DNRL，则形状为 [每个 batch 中被采样到子图中的节点个数,  2 *self.dim]
            # print("all_mask_shape=", all_mask.shape)
            # print("pairwise_feats shape=", pairwise_feats.shape)
            # print("pes_shape=", pes.shape)
            for l in range(self.num_layers):
                # 论文注释：这里的 pairwise_feats 是 s(a,b)
                # 实际使用的 num_layers 为 1，更新后 pairwise_feats 形状变为 [batchsize, self.dim]
                pairwise_feats, att_weights = self.att_layers[l](all_mask, pairwise_feats, X_node, pes, None, return_weights)
                
                # 如果这里改成 SubGraphormer，则为
                # h_graphormer, (u_pos, v_pos) = self.graphormer(self.data, subg_mask, batch[0], batch[1])
                # 设法将输出的 h_u, h_v 和 h_CLS 形状映射为 [batchsize, self.dim] 即可
            
            num_cns, num_1hop, num_non1hop, num_neighbors = self.get_structure_cnts(batch, cn_info, onehop_info, non1hop_info, test_set=test_set) 
            # 论文注释：就是把其他个数 concat 进去，即论文最后的部分。
            if num_non1hop is not None:
                pairwise_feats = torch.cat((pairwise_feats, num_cns, num_1hop, num_non1hop, num_neighbors), dim=-1)
            else:
                pairwise_feats = torch.cat((pairwise_feats, num_cns, num_1hop, num_neighbors), dim=-1)
        
        # 拼接完以后，pairwise_feats 形状为 [batchsize, pairwise_dim]，其中 pairwise_dim = self.dim * train_args['num_heads'] + count_dim
        pairwise_feats = self.pairwise_lin(pairwise_feats)
        return pairwise_feats, att_weights

    

    def get_pos_encodings(self, cn_info, onehop_info=None, non1hop_info=None, drnl_info=None):
        """
        Ensure symmetric by making `enc = g(a, b) + g(b, a)`

        Returns:
        --------
        torch.Tensor
            Concatenated encodings for cn and 1-hop
        """
        # 各个指标的 info 就是 compute_node_mask 函数算出来的三个尺度的信息，其中有三个元素
        # info[0] 是下标的稠密矩阵。# info[1] 是 ppr(a,u)， info[2] 是 ppr(b,u)，形状为 (子图节点个数)
        # 如果是 drnl_info，则只有两个元素
        
        # n 为整个 batch 中被采样到子图中的节点个数
        # 下面的 torch.stack((cn_info[1], cn_info[2]) 堆叠后形状为 (2, n), 加上 t() 转置为 (n, 2)，对应原文就是 CN 中的每个节点的 ppr(a,u) 和 ppr(b,u) 拼接起来
        # ppr_encoder_cn 是一个 MLP
        # 出结果的 cn_a 就是论文里的 rpe(a,u)，形状为 (n, self.dim)

        cn_a = self.ppr_encoder_cn(torch.stack((cn_info[1], cn_info[2])).t())
        # print("cn_a", cn_a.shape)
        cn_b = self.ppr_encoder_cn(torch.stack((cn_info[2], cn_info[1])).t())
        # 论文注释，这里就是保证顺序不变性的那个
        cn_pe = cn_a + cn_b


        if onehop_info is None:
            return cn_pe

        onehop_a = self.ppr_encoder_onehop(torch.stack((onehop_info[1] , onehop_info[2])).t())
        onehop_b = self.ppr_encoder_onehop(torch.stack((onehop_info[2], onehop_info[1])).t())
        onehop_pe = onehop_a + onehop_b


        if non1hop_info is None:
            return torch.cat((cn_pe, onehop_pe), dim=0)

        non1hop_a = self.ppr_encoder_non1hop(torch.stack((non1hop_info[1] , non1hop_info[2])).t())
        non1hop_b = self.ppr_encoder_non1hop(torch.stack((non1hop_info[2] , non1hop_info[1])).t())
        non1hop_pe = non1hop_a + non1hop_b

        if drnl_info is not None:
            drnl_pe = self.drnl_encoder(drnl_info[1].unsqueeze(1))
            # drnl position encoding 形状为 (n, self.dim)
            rpe_pe = torch.cat((cn_pe, onehop_pe, non1hop_pe), dim=0)
            # 返回结果形状应为 (n, 2 * self.dim)
            return torch.cat((drnl_pe, rpe_pe), dim=1)
        else:
            return torch.cat((cn_pe, onehop_pe, non1hop_pe), dim=0)


    def compute_node_mask(self, batch, test_set, adj):
        """
        Get mask based on type of node

        When mask_type != "cn", also return the ppr vals for both the 
        source and target

        NOTE:
            1. Adj used here has no edge weights. Only 0/1!!!
            2. Adj must already be coalesced for this to work correctly!!!
            3. Pos Edges in batch must be masked.
        """
        if adj is None:
            adj = self.get_adj(test_set, mask=True)

        # batch[0] 和 batch[1] 是目标节点对的下标，下面通过 index_select，借助邻接矩阵，找出所有与目标节点 a 或 b 有连接的那些节点的下标，得到一个长度为节点个数的向量，表示了各个节点
        # 此时 src_adj 为与目标节点 a 相连的节点的下标，形状应为 (batchsize, 所有节点个数)，1 为真，0 为假
        src_adj = torch.index_select(adj, 0, batch[0])
        tgt_adj = torch.index_select(adj, 0, batch[1])

        if self.mask == "cn":
            # 1 when CN, 0 otherwise
            pair_adj = src_adj * tgt_adj
        else:
            # Equals: {0: ">1-Hop", 1: "1-Hop (Non-CN)", 2: "CN"}
            pair_adj = src_adj + tgt_adj
        # 此时 pair_adj 记录了不同类型子图的信息，形状为 (batchsize, 所有节点个数)
        # pair_adj = self.subgraph_mask
        pair_ix, node_type, src_ppr, tgt_ppr = self.get_ppr_vals(batch, pair_adj, test_set)
        # pair_ix 形状为 (2, n)，其中每列存储了 ppr 非零节点的二维下标（即 (batchsize, 节点个数)），由两个数字组成，表示了在原来的矩阵中对应的位置

        # 接下来根据 PPR 值做筛选从而选出子图
        cn_filt_cond = (src_ppr >= self.thresh_cn) & (tgt_ppr >= self.thresh_cn)
        onehop_filt_cond = (src_ppr >= self.thresh_1hop) & (tgt_ppr >= self.thresh_1hop)
        # print("cn_filt_cond", cn_filt_cond.shape)
        # filt_cond 为筛选后的布尔值，形状为 (n)
    
        if self.mask != "cn":
            filt_cond = torch.where(node_type == 1, onehop_filt_cond, cn_filt_cond)
        else:
            filt_cond = torch.where(node_type == 0, onehop_filt_cond, cn_filt_cond)

        pair_ix, node_type = pair_ix[:, filt_cond], node_type[filt_cond]
        # print("pair_ix 2 shape=", pair_ix.shape)
        src_ppr, tgt_ppr = src_ppr[filt_cond], tgt_ppr[filt_cond]

        # >1-Hop mask is gotten separately
        if self.mask == "all":
            non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr = self.get_non_1hop_ppr(batch, test_set=test_set)

        # Dropout
        if self.training and self.att_drop > 0:
            pair_ix, src_ppr, tgt_ppr, node_type = self.drop_pairwise(pair_ix, src_ppr, tgt_ppr, node_type)
            if self.mask == "all":
                non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr, _ = self.drop_pairwise(non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr)

        # Separate out CN and 1-Hop
        if self.mask != "cn":
            cn_ind = node_type == 2
            # print("cn_ind shape=", cn_ind.shape)
            cn_ix, cn_src_ppr, cn_tgt_ppr = pair_ix[:, cn_ind], src_ppr[cn_ind], tgt_ppr[cn_ind]

            one_hop_ind = node_type == 1
            # print("one_hop_ind shape=", one_hop_ind.shape)
            onehop_ix, onehop_src_ppr, onehop_tgt_ppr = pair_ix[:, one_hop_ind], src_ppr[one_hop_ind], tgt_ppr[one_hop_ind]
        
        # 这里的 ix 均为稠密矩阵形式，表示了 batch 和节点下标
        # print("\n\ncn_in=", cn_ix.shape)
        # print("onehop_ix", onehop_ix.shape)
        # print("non1hop_ix", non1hop_ix.shape)
        if self.mask == "cn":
            return (pair_ix, src_ppr, tgt_ppr), None, None
        elif self.mask == "1-hop":
            return (cn_ix, cn_src_ppr, cn_tgt_ppr), (onehop_ix, onehop_src_ppr, onehop_tgt_ppr), None
        else:
            return (cn_ix, cn_src_ppr, cn_tgt_ppr), (onehop_ix, onehop_src_ppr, onehop_tgt_ppr), (non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr)


    def get_ppr_vals(self, batch, pair_diff_adj, test_set=False):
        """
        Get the src and tgt ppr vals

        `pair_diff_adj` specifies type of nodes we select

        input:
            batch: 形状为（2, batch_size）的张量。每一行分别表示源节点和目标节点的下标。
            pair_diff_adj: 形状为（batch_size, 所有节点个数）的张量，用于指定选择节点所属的子图类型。类别如下：
                - `0`: 表示">1-Hop"的节点
                - `1`: 表示"1-Hop (Non-CN)"的节点
                - `2`: 表示"CN"的节点
            test_set: 布尔值，默认为`False`。用于指示是否使用测试集来计算PPR值。
        """
        # Additional terms for also choosing scores when ppr=0
        # Multiplication removes any values for nodes not in batch
        # Addition then adds offset to ensure we select when ppr=0
        # All selected scores are +1 higher than their true val

        # ppr 矩阵，存储所有节点对的PPR值，形状为 (节点个数, 节点个数) 。
        # 考虑到对于同一个数据集，不同的链路预测样本不影响 ppr 值，因此这里不需要考虑 batchsize
        # ppr[i][j]表示从节点i到节点j的PPR值。
        ppr = self.get_ppr(test_set)

        # torch.index_select(ppr, 0, batch[0]) 这一步是从 ppr 这个矩阵选择源节点对应的行，读出对应的数据
        # 添加偏移量，确保后续能区分 PPR 值为 0 的情况
        # src_ppr_adj 表示了当前 batch 的每个源节点到其他节点的调整后 PPR 值，同理 tgt_ppr_adj 是目标节点到其他节点的调整后 PPR 值。形状均为 (batchsize, 节点个数)
        
        src_ppr_adj = torch.index_select(ppr, 0, batch[0]) * pair_diff_adj + pair_diff_adj
        tgt_ppr_adj = torch.index_select(ppr, 0, batch[1]) * pair_diff_adj + pair_diff_adj

        # Can now convert ppr scores to dense
        # 这里有一部分稀疏矩阵转换为稠密矩阵的计算。
        # ppr_ix 为 src_ppr_adj 中非零元素的坐标构成的矩阵。
        # 假如图中有 n 个节点到源节点的 ppr 不为零，则 ppr_ix 形状为 (2, n)，其中每列存储了非零节点的二维下标（即 (batchsize, 节点个数)），由两个数字组成，表示了在原来的矩阵中对应的位置
        # 相应的值则被存进了 src_ppr，为一维向量，长度为 (n)
        ppr_ix  = src_ppr_adj.coalesce().indices()
        src_ppr = src_ppr_adj.coalesce().values()
        tgt_ppr = tgt_ppr_adj.coalesce().values()
        # print("ppr_ix:", ppr_ix.shape)
        # print("src_ppr:", src_ppr.shape)

        # If one is 0 so is the other
        # NOTE: Should be few to no nodes here
        # src_ppr = src_ppr[src_ppr != 0]
        # tgt_ppr = tgt_ppr[tgt_ppr != 0]
        # ppr_ix = ppr_ix[:, src_ppr != 0]
        
        # TODO: Needed due to a bug in recent torch versions 
        # see here for more - https://github.com/pytorch/pytorch/issues/114529
        # note that if one is 0 so is the other
        zero_vals = (src_ppr != 0) 
        src_ppr = src_ppr[zero_vals]
        tgt_ppr = tgt_ppr[tgt_ppr != 0]
        ppr_ix = ppr_ix[:, zero_vals]
        # print("src_ppr 2:", src_ppr.shape)
        # print("ppr_ix 2:", ppr_ix.shape)

        pair_diff_adj = pair_diff_adj.coalesce().values()
        # print("pair_diff_adj 2:", pair_diff_adj.shape)
        pair_diff_adj = pair_diff_adj[src_ppr != 0]
        # print("pair_diff_adj 3:", pair_diff_adj.shape)

        # Remove additional +1 from each ppr val
        src_ppr = (src_ppr - pair_diff_adj) / pair_diff_adj
        tgt_ppr = (tgt_ppr - pair_diff_adj) / pair_diff_adj

        return ppr_ix, pair_diff_adj, src_ppr, tgt_ppr


    def drop_pairwise(self, node_ix, src_ppr=None, tgt_ppr=None, node_indicator=None):
        """
        Drop nodes used in pairwise info
        """
        num_indices = math.ceil(node_ix.size(1) * (1-self.att_drop))
        indices = torch.randperm(node_ix.size(1))[:num_indices]
        node_ix = node_ix[:, indices]

        if src_ppr is not None:
            src_ppr = src_ppr[indices]
        if tgt_ppr is not None:
            tgt_ppr = tgt_ppr[indices]
        if node_indicator is not None:
            node_indicator = node_indicator[indices]

        return node_ix, src_ppr, tgt_ppr, node_indicator
    

    def get_structure_cnts(self, batch, cn_info, onehop_info, non1hop_info=None, test_set=None):
        """
        Counts for CNs, 1-Hop, and >1-Hop
        """
        # 论文注释：根据三种不同位置关系，给不同阈值进行筛选并计算个数
        num_cns = self.get_count(cn_info[0], batch, test_set)            
        num_1hop = self.get_num_ppr_thresh(batch, onehop_info[0], onehop_info[1], 
                                           onehop_info[2], test_set=test_set)

        num_ppr_ones = self.get_num_ppr_thresh(batch, onehop_info[0], onehop_info[1], 
                                               onehop_info[2], thresh=0, test_set=test_set)
        num_neighbors = num_cns + num_ppr_ones

        if non1hop_info is None:
            return num_cns, num_1hop, None, num_neighbors
        else:
            num_non1hop = self.get_count(non1hop_info[0], batch, test_set)
            return num_cns, num_1hop, num_non1hop, num_neighbors


    def get_num_ppr_thresh(self, batch, onehop_mask, src_ppr, tgt_ppr, test_set=False, thresh=None):
        """
        Get # of nodes where ppr(a, v) >= thresh & ppr(b, v) >= thresh

        When src_ppr is None just get srabda
        """
        if thresh is None:
            thresh = self.thresh_1hop

        weight = torch.ones(onehop_mask.size(1), device=onehop_mask.device)

        ppr_above_thresh = (src_ppr >= thresh) & (tgt_ppr >= thresh)
        num_ppr = scatter(ppr_above_thresh.float() * weight, onehop_mask[0].long(), dim=0, dim_size=batch.size(1), reduce="sum")
        num_ppr = num_ppr.unsqueeze(-1)

        return num_ppr


    def get_count(self, node_mask, batch, test_set):
        """
        # of CNs for each sample in batch 
        """
        weight = torch.ones(node_mask.size(1), device=node_mask.device)

        num_cns = scatter(weight, node_mask[0].long(), dim=0, dim_size=batch.size(1), reduce="sum")
        num_cns = num_cns.unsqueeze(-1)

        return num_cns


    def get_adj(self, test_set=False, mask=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        # 林子垚：从数据中读邻接矩阵或者是掩码矩阵
        suffix = "mask" if mask else "t"
        if test_set:
            return self.data[f'full_adj_{suffix}']
        
        return self.data[f'adj_{suffix}']

    def get_ppr(self, test_set=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        if test_set and 'ppr_test' in self.data:
            return self.data[f'ppr_test']
        
        return self.data[f'ppr']

    def get_degree(self, test_set=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        if test_set and 'degree_test' in self.data:
            return self.data[f'degree_test']
        
        return self.data[f'degree']
    

    def agg_by_weight(self, batch, X, weight_ix, weight_vals=None):
        """
        Perform a weighted sum by weights for each node in batch
        """
        batch_num = weight_ix[0]   # Corresponding entry for node

        if weight_vals is not None:
            # weighted_hids = weight_vals.unsqueeze(-1) * X[ppr_ix[1]]
            weighted_hids = weight_vals * X[weight_ix[1]]
        else:
            weighted_hids = X[weight_ix[1]]

        output = scatter(weighted_hids, batch_num, dim=0, dim_size=batch.size(1), reduce="sum")

        return output

    def get_non_1hop_ppr(self, batch, test_set=False):
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

    def graphormer(self, x):
        """
        基于 Graphormer 的思路，用输入的 CN，PPR，DNRL 等做启发式指标计算来算 key 和 query

        Input:
        x: [batch_size, 3 * dim] 为包含了其他节点信息等表征

        Output:
        output: [batch_size, dim] 为最终的输出
        """

