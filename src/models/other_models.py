import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add

    


class GCN(torch.nn.Module):
    """
    GCN Model
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers,
        dropout,
        residual=False,
        cached=False,
        normalize=True,
        layer_norm=True,
        relu=True,
    ):
        super(GCN, self).__init__()

        self.relu = relu
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            hidden_channels = out_channels
            
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=normalize))
        
        if layer_norm:
            self.lns = torch.nn.ModuleList()
            self.lns.append(nn.LayerNorm(hidden_channels))
        else:
            self.lns = None

        if num_layers > 1:    
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=normalize))
                self.lns.append(nn.LayerNorm(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached, normalize=normalize))
            if layer_norm:
                self.lns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.residual = residual


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):
        """
        Propagation
        """
        for i, conv in enumerate(self.convs):
            xi = conv(x, adj_t)
            xi = self.lns[i](xi) if self.lns is not None else xi
            xi = F.dropout(xi, p=self.dropout, training=self.training)
            xi = F.relu(xi) if self.relu else xi

            if self.residual and x.shape[-1] == xi.shape[-1]:
                x = x + xi
            else:
                x = xi

        return x



class MLP(nn.Module):
    """
    L Layer MLP
    """
    def __init__(
        self, 
        num_layers,
        in_channels, 
        hid_channels, 
        out_channels, 
        drop=0,
        norm="layer",
        sigmoid=False,
        bias=True
    ):
        super().__init__()
        self.dropout = drop
        self.sigmoid = sigmoid

        if norm == "batch":
            self.norm = nn.BatchNorm1d(hid_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(hid_channels)
        else:
            self.norm = None

        self.linears = torch.nn.ModuleList()

        if num_layers == 1:
            self.linears.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            self.linears.append(nn.Linear(in_channels, hid_channels, bias=bias))
            for _ in range(num_layers-2):
                self.linears.append(nn.Linear(hid_channels, hid_channels, bias=bias))
            self.linears.append(nn.Linear(hid_channels, out_channels, bias=bias))
    

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        
        if self.norm is not None:
            self.norm.reset_parameters()


    def forward(self, x):
        """
        Forward Pass
        """
        for i, lin in enumerate(self.linears[:-1]):
            x = lin(x)
            x = self.norm(x) if self.norm is not None else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linears[-1](x)
        x = x.squeeze(-1)

        return torch.sigmoid(x) if self.sigmoid else x



class mlp_score(torch.nn.Module):
    """
    MLP score function
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers,
        dropout=0
    ):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze(-1)




class PairMLP(nn.Module):
    def __init__(
        self, 
        num_layers: int,
        in_channels: int,  # 单个节点的特征维度（u_emb和v_emb的输入维度）
        hid_channels: int, 
        out_channels: int,  # 最终输出维度（self.dim）
        drop: float = 0.0,
        norm: str = "layer",
        sigmoid: bool = False,
        bias: bool = True,
        merge_method: str = "concat",  # 新增参数：合并u_emb和v_emb的方式
    ):
        super().__init__()
        assert num_layers >= 1, "至少需要1个线性层"
        assert merge_method in ["concat", "sum", "mul", "avg"], "合并方式无效"
        
        self.merge_method = merge_method
        self.sigmoid = sigmoid
        
        # 根据合并方式调整输入维度
        if self.merge_method == "concat":
            merged_dim = 2 * in_channels  # 拼接后维度翻倍
        else:
            merged_dim = in_channels  # 其他方式维度不变
            
        # 构建隐藏层
        layers = []
        for i in range(num_layers - 1):  # 前n-1层为隐藏层
            in_dim = merged_dim if i == 0 else hid_channels
            layers.append(nn.Linear(in_dim, hid_channels, bias=bias))
            
            # 归一化层
            if norm == "layer":
                layers.append(nn.LayerNorm(hid_channels))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hid_channels))
                
            layers.append(nn.ReLU())  # 激活函数
            layers.append(nn.Dropout(drop))
        
        # 最后一层（输出层）
        layers.append(nn.Linear(hid_channels, out_channels, bias=bias))
        self.mlp = nn.Sequential(*layers)
        
    def _merge_embeddings(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        """合并u和v的特征"""
        if self.merge_method == "concat":
            return torch.cat([u_emb, v_emb], dim=-1)  # [batch, 2*in_channels]
        elif self.merge_method == "sum":
            return u_emb + v_emb
        elif self.merge_method == "mul":
            return u_emb * v_emb
        elif self.merge_method == "avg":
            return (u_emb + v_emb) / 2.0
        else:
            raise ValueError(f"不支持的合并方式: {self.merge_method}")

    def forward(
        self, 
        u_emb: torch.Tensor,  # 形状 [batch, in_channels]
        v_emb: torch.Tensor   # 形状 [batch, in_channels]
    ) -> torch.Tensor:
        merged = self._merge_embeddings(u_emb, v_emb)  # 合并
        out = self.mlp(merged)  # 通过MLP
        
        if self.sigmoid:
            return torch.sigmoid(out)  # 输出概率
        return out  # 形状 [batch, out_channels]