import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import math
from torch import Tensor
from typing import Optional, Union

from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros


class LinkTransformerLayer(nn.Module):
    """
    Layer for models.LinkTransformer
    应该就是原文计算 s(a,b) 的过程
    """
    def __init__(
        self, 
        dim, 
        train_args,
        concat=True,  # TODO: !!!!
        out_dim=None,
        node_dim=None 
    ):
        super().__init__()

        self.dropout = train_args.get("dropout", 0) 
        out_dim = dim if out_dim is None else out_dim

        self.att = LinkAttention(dim, out_dim, train_args, flow="target_to_source", 
                                 concat=concat, node_dim=node_dim)
        self.post_att_norm = nn.LayerNorm(out_dim * train_args['num_heads'] if concat else out_dim)
    

    def forward(self, edge_index, edge_x, node_x, pe_enc=None, deg_enc=None, return_weights=False):
        """
        Encode the edge features

        Parameters:
        ----------
            edge_index: torch.Tensor
                Edge Index based on node mask. 1st Dim = edge, 2nd dim = Node
            edge_x: torch.Tensor
                Size (BS, dim). Contains the representations of all edges in the batch.
                Corresponds to attention query.
            node_key: torch.Tensor
                Size (|V|, dim). Contains representations of all nodes corresponding to attention key
            pe_scalars: torch.Tensor
                Same size as edge_index.size(0). Holds scalar corresponding to DRNL between edges and nodes
            return_weights: False
                Return att weights
        
        Returns:
        -------
        tuple 
            torch.Tensor
                Encoded representations of size (BS x dim)
            torch.Tensor (Optional)
                Att weights (BS, |V|) or None
        """
        if not return_weights:
            out = self.att(edge_x, node_x, edge_index, pe_enc, deg_enc)
            att_weights = None
        else:
            ##############################
            # NOTE: Get att weights. For debugging purposes
            # Weight shape is (num_heads, edge_index.size(1))
            # Average for ease
            out, weights = self.att(edge_x, node_x, edge_index, pe_enc, deg_enc, return_att_weights=True)
            weights = weights.mean(axis=0)
            att_weights = torch.stack((edge_index[0], weights))
    
        # TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 论文注释：多头注意力
        out = self.post_att_norm(out)

        out = F.dropout(out, p=self.dropout, training=self.training)

        return out, att_weights





class LinkAttention(MessagePassing):
    """
    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        train_args: dict, 
        concat: bool = True,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        # edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = True,
        node_dim=None,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = train_args['num_heads']
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = train_args.get('att_drop', 0)
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.lin_edge = None  # No edge embedding

        # 为啥这里是 2
        out_dim = 3

        if node_dim is None:
            node_dim = in_channels * out_dim
            # node_dim = in_channels + 1
        else:
            node_dim = node_dim * out_dim
        
        self.lin_l = Linear(in_channels, self.heads * out_channels, bias=bias, weight_initializer='glorot')
        self.lin_r = Linear(node_dim, self.heads * out_channels, bias=bias, weight_initializer='glorot')

        att_out = out_channels
        self.att = Parameter(torch.Tensor(1, self.heads, att_out))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)



    def forward(
        self, 
        edge_x,
        node_x,
        edge_index,
        pos_enc = None,
        deg_enc = None,
        return_att_weights = None
    ):
        """
        Runs the forward pass of the module.
        """
        out = self.propagate(edge_index, x=(edge_x, node_x), pe_att=pos_enc, deg_enc=deg_enc, size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_att_weights:
            return out, alpha.T
        else:
            return out


    def message(
        self, 
        x_i: Tensor,    # Edge
        x_j: Tensor,    # Node (Trust me!!!)
        pe_att: Tensor, 
        # drnl: Tensor,
        deg_enc: Tensor,
        index: Tensor, 
        ptr,
        size_i: Optional[int]
    ):
        H, C = self.heads, self.out_channels
        # print("x_j=", x_j.shape)
        # print("pe_att=", pe_att.shape)
        if pe_att is not None:
            # 论文注释：x_j 就是 h(a,b,u)= [h_u||rpe(a,b,u)]，只不过这里是对所有节点进行的，维度不一样，下同。
            x_j = torch.cat((x_j, pe_att), dim=-1)
        # print("x_j 2=", x_j.shape)

        # 把 h(a,b,u) 拼接后的结果过一个线性层
        x_j = self.lin_r(x_j).view(-1, H, C)
        # 论文注释：下面进行的应该就是 GATv2 注意力机制，有点看不通
        # e=(a, b) attending to v
        
        # x_a * x_v + x_b * x_v
        e1, e2 = x_i.chunk(2, dim=-1) 
        e1 = self.lin_l(e1).view(-1, H, C)
        e2 = self.lin_l(e2).view(-1, H, C)
        x = x_j * (e1 + e2)
        
        x = F.leaky_relu(x, self.negative_slope)

        # 论文注释：x 就是 tilde w(a,b,u)
        alpha = (x * self.att).sum(dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        # 论文注释：做完 softmax 就是 w(a,b,u)
        self._alpha = alpha

        # 最后 w(a,b,u) 和 h(a,b,u) 逐元素相乘得到 s(a,b,u)
        return x_j * alpha.unsqueeze(-1)


class GraphTransformer(nn.Module):
    """
    这是一个 Graph Transformer ，基于某个子图的所有节点的 token 计算相互注意力，并且做更新。

    输入：
    h_token: torch.Tensor
        Size (n, self.dim). 包含了当前 batch 中所有被采样节点的 token。n 为当前 batch 中所有处于子图中的节点个数。
    batch: torch.Tensor
        Size (2, batch_size). 包含了当前 batch 中所有样本的目标节点对在全图中的下标信息，其中 h[0] 是源节点 a，h[1] 是目标节点 b
    subg_mask: torch.Tensor
        Size (2, n). 压缩后的二维 COO 坐标形式的稠密子图掩码矩阵，每列格式为 (sample_idx, node_idx)
    输出：
    h_a: torch.Tensor
        Size (batch_size，self.dim). 为更新后源节点 a 的 token
    h_b: torch.Tensor
        Size (batch_size，self.dim). 为更新后目标节点 b 的 token
    h_CLS: torch.Tensor
        Size (batch_size，self.dim). 为更新后 CLS 节点的 token
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

        # LayerNorm and FeedForward layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, h_token, batch, subg_mask):
        # Step 1: Split tokens based on subg_mask
        sample_idx, node_idx = subg_mask
        batch_size = batch.size(1)
        subgraph_tokens = []
        for i in range(batch_size):
            mask = sample_idx == i
            subgraph_tokens.append(h_token[node_idx[mask]])
        
        # Step 2: Add CLS token to each subgraph
        cls_tokens = []
        padded_tokens = []
        max_nodes = max(len(tokens) for tokens in subgraph_tokens)
        for tokens in subgraph_tokens:
            cls_token = tokens.mean(dim=0, keepdim=True)  # MeanPool for CLS initialization
            cls_tokens.append(cls_token)
            tokens_with_cls = torch.cat([cls_token, tokens], dim=0)
            pad_size = max_nodes + 1 - tokens_with_cls.size(0)
            padded_tokens.append(F.pad(tokens_with_cls, (0, 0, 0, pad_size)))  # Padding
        
        padded_tokens = torch.stack(padded_tokens)  # Shape: (batch_size, max_nodes + 1, dim)
        cls_tokens = torch.cat(cls_tokens, dim=0)  # Shape: (batch_size, dim)

        # Step 3: Compute attention
        padded_tokens = padded_tokens.transpose(0, 1)  # Shape: (max_nodes + 1, batch_size, dim)
        attn_output, _ = self.attention(padded_tokens, padded_tokens, padded_tokens)
        attn_output = attn_output.transpose(0, 1)  # Shape: (batch_size, max_nodes + 1, dim)

        # Step 4: Apply LayerNorm and FeedForward
        attn_output = self.norm1(attn_output + padded_tokens.transpose(0, 1))
        output = self.norm2(attn_output + self.ffn(attn_output))

        # Step 5: Extract updated tokens
        h_a = output[:, 1, :]  # Updated source node tokens
        h_b = output[:, 2, :]  # Updated target node tokens
        h_CLS = output[:, 0, :]  # Updated CLS tokens

        return h_a, h_b, h_CLS
