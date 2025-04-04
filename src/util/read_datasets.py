import os 
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_
import scipy.sparse as sp

import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, degree

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset

from util.calc_ppr_scores import get_ppr
from util.utils import torch_sparse_tensor_to_sparse_mx, sparse_mx_to_torch_sparse_tensor

import networkx as nx
import hashlib
import scipy.sparse as sp
from time import time
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "dataset")
HEART_DIR = os.path.join(DATA_DIR, "heart")


def read_data_ogb(args, global_logger, device):
    """
    Read data for OGB datasets
    """
    data_obj = {
        "dataset": args.data_name,
    }

    print("Loading all data...")

    dataset = PygLinkPropPredDataset(name=args.data_name)
    data = dataset[0].to(device)
    split_edge = dataset.get_edge_split()

    if "collab" in args.data_name:
        data, split_edge = filter_by_year(data, split_edge)
        data = data.to(device)

    data_obj['num_nodes'] = data.num_nodes
    edge_index = data.edge_index

    if args.data_name != 'ogbl-citation2':
        data_obj['train_pos'] = split_edge['train']['edge'].to(device)
        data_obj['valid_pos'] = split_edge['valid']['edge'].to(device)
        data_obj['valid_neg'] = split_edge['valid']['edge_neg'].to(device)
        data_obj['test_pos'] = split_edge['test']['edge'].to(device)
        data_obj['test_neg'] = split_edge['test']['edge_neg'].to(device)
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        data_obj['train_pos'] = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1).to(device)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        data_obj['valid_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['valid_neg'] = split_edge['valid']['target_node_neg'].to(device) 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['test_neg'] = split_edge['test']['target_node_neg'].to(device)

    # Overwrite Val/Test pos sample for ogbl-ppa under HeaRT
    if args.heart and "ppa" in args.data_name:
        with open(f'{HEART_DIR}/{args.data_name}/valid_samples_index.pt', "rb") as f:
            val_pos_ix = torch.load(f)
        with open(f'{HEART_DIR}/{args.data_name}/test_samples_index.pt', "rb") as f:
            test_pos_ix = torch.load(f)

        data_obj['valid_pos'] = data_obj['valid_pos'][val_pos_ix, :]
        data_obj['test_pos'] = data_obj['test_pos'][test_pos_ix, :]

    # Test train performance without evaluating all test samples
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]

    if hasattr(data, 'x') and data.x is not None:
        data_obj['x'] = data.x.to(device).to(torch.float)
    else:
        data_obj['x'] =  torch.nn.Parameter(torch.zeros(data_obj['num_nodes'], args.dim).to(device))
        xavier_uniform_(data_obj['x'])

    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight.to(torch.float)
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    else:
        edge_weight = torch.ones(edge_index.size(1)).to(device).float()
    
    data_obj['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight.squeeze(-1), [data.num_nodes, data.num_nodes]).to(device)

    # Start Norm
    if args.mat_prop >= 0:
        print("Getting Normalized Adj...")
        # 按照原文创建 torch_Sparse 类型
        adj = SparseTensor.from_edge_index(edge_index, edge_weight.squeeze(-1), [data.num_nodes, data.num_nodes])
        # 转为 torch.sparse coo 类型（参见 torch_Sparse 的 GitHub 相关代码）
        adj = adj.to_torch_sparse_coo_tensor()
        # 再变成 scipy 稀疏矩阵
        adj = torch_sparse_tensor_to_sparse_mx(adj)
        adj = adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')
        A = adj.dot(D1)
        A = D2.dot(A)
        adj = sparse_mx_to_torch_sparse_tensor(A).to(device)
        print("Done!")

        data_obj['norm_adj'] = adj

    # Needed since directed graph
    # 林子垚：因为 "ogbl-citation2" 是有向图，所以需要单独处理。具体来说，需要通过 to_symmetric 转为无向图，然后用 coalesce 合成重复边。此时成为了为无向图
    # adj_mask 表示图中节点之间的连接关系，to_torch_sparse_coo_tensor 是一种稀疏格式
    if args.data_name == 'ogbl-citation2': 
        data_obj['adj_t'] = data_obj['adj_t'].to_symmetric().coalesce()
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
    else:
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()        
    
    # Don't use edge weight. Only 0/1. Not needed for masking
    # 林子垚：将掩码转换为布尔类型，表示连接的存在与否
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()

    if args.use_val_in_test:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index).to(device)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data['full_edge_index'] = full_edge_index.to(device)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float, device=device)
        val_edge_weight = torch.cat([edge_weight, val_edge_weight], 0).view(-1)
        data_obj['full_adj_t'] = SparseTensor.from_edge_index(full_edge_index, val_edge_weight, [data.num_nodes, data.num_nodes]).to(device)

        # Don't use edge weight. Only 0/1. Not needed for masking
        data_obj['full_adj_mask'] = data_obj['full_adj_t'].to_torch_sparse_coo_tensor()
        data_obj['full_adj_mask'] = data_obj['full_adj_mask'].coalesce().bool().int()
    else:
        data_obj['full_adj_t'] = data_obj['adj_t']
        data_obj['full_adj_mask'] = data_obj['adj_mask']

    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes']).to(device)
    if args.use_val_in_test:
        data_obj['degree_test'] = degree(full_edge_index[0], num_nodes=data_obj['num_nodes']).to(device)

    ### Load PPR matrix
    print("Reading PPR...", flush=True)
    data_obj['ppr'] = get_ppr(args.data_name, edge_index, data['num_nodes'],
                              0.15, args.eps, False).to(device)  

    if args.use_val_in_test:
        data_obj['ppr'] = get_ppr(args.data_name, data['full_edge_index'], data['num_nodes'],
                                0.15, args.eps, True).to(device)  
    else:
        data_obj['ppr_test'] = data_obj['ppr']

    # Overwrite standard negatives
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data_obj['valid_neg'] = torch.from_numpy(neg_valid_edge).to(device)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data_obj['test_neg'] = torch.from_numpy(neg_test_edge).to(device)

        # For DDI, val/test takes a long time so only use a subset of val
        if "ddi" in args.data_name:
            num_sample = data_obj['valid_pos'].size(0) // 4
            idx = torch.randperm(data_obj['valid_pos'].size(0))[:num_sample].to(device)
            data_obj['valid_pos'] = data_obj['valid_pos'][idx]
            data_obj['valid_neg'] = data_obj['valid_neg'][idx]
            data_obj['train_pos_val'] = data_obj['train_pos_val'][idx]

    return data_obj




def read_data_planetoid(args, device):
    """
    Read all data for the fixed split. Returns as dict
    """
    data_name = args.data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        path = os.path.join(DATA_DIR, data_name, f"{split}_pos.txt")
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('# of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:
        path = os.path.join(DATA_DIR, data_name, f"{split}_neg.txt")

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)

            if split == 'valid': 
                valid_neg.append((sub, obj))               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))
          
    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(os.path.join(DATA_DIR, data_name, "gnn_feature"))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {"dataset": args.data_name}
    data['edge_index'] = edge_index.to(device)
    data['num_nodes'] = num_nodes

    data['train_pos'] = train_pos_tensor.to(device)
    # print("Num of train positive sample:", data['train_pos'].shape)
    data['train_pos_val'] = train_val.to(device)

    data['valid_pos'] = valid_pos.to(device)
    data['valid_neg'] = valid_neg.to(device)
    data['test_pos'] = test_pos.to(device)
    data['test_neg'] = test_neg.to(device)

    data['x'] = feature_embeddings.to(device)

    data['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes]).to(device)
    data['full_adj_t'] = data['adj_t'].to(device)

    # 处理邻接矩阵，用于后续的邻接矩阵传播以及 DRNL 计算
    # 按照原文创建 torch_Sparse 类型
    adj = SparseTensor.from_edge_index(edge_index, edge_weight.squeeze(-1), [num_nodes, num_nodes]).to(device)
    # 转为 torch.sparse coo 类型（参见 torch_Sparse 的 GitHub 相关代码）
    adj = adj.to_torch_sparse_coo_tensor()
    # 再变成 scipy 的 coo 稀疏矩阵
    coo_adj = torch_sparse_tensor_to_sparse_mx(adj)
    data['coo_adj'] = coo_adj

    # Normalize Adj
    if args.mat_prop >= 0:
        adj = coo_adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')
        A = adj.dot(D1)
        A = D2.dot(A)
        adj = sparse_mx_to_torch_sparse_tensor(A).to(device)

        data['norm_adj'] = adj


    data['adj_mask'] = data['adj_t'].to_torch_sparse_coo_tensor()
    data['full_adj_mask'] = data['adj_mask'] = data['adj_mask'].coalesce()

    ### Degree of nodes
    data['degree'] = degree(data['edge_index'][0], num_nodes=data['num_nodes']).to(device)

    ### Load PPR Matrix
    data['ppr'] = get_ppr(args.data_name, data['edge_index'], data['num_nodes'],
                          0.15, args.eps, False).to(device)
    data['ppr_test'] = data['ppr']
    # Overwrite standard negative
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data['valid_neg'] = torch.from_numpy(neg_valid_edge)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data['test_neg'] = torch.from_numpy(neg_test_edge)
    
    # data = compute_all_pairs_heuristics(data, use_len_spd=True, use_num_spd=True, use_cnb_jac=True, use_cnb_aa=True, use_cnb_ra=True)
    return data



    
def filter_by_year(data, split_edge, year=2007):
    """
    From BUDDY code

    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def compute_heuristics(data, use_len_spd=True, use_num_spd=True, 
                      use_cnb_jac=True, use_cnb_aa=True, use_cnb_ra=True):
    print("Computing heuristics...")
    # 创建无向图
    # print("edge shape=", data['edge_index'].shape)
    num_nodes = data['x'].shape[0]
    edge_index = data['edge_index'].cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T)
    
    # 初始化存储字典
    heuristics = {}
    
    # 需要处理的边集合
    splits = {
        'train_pos': data['train_pos'],
        'valid_pos': data['valid_pos'],
        'test_pos': data['test_pos'],
        'valid_neg': data['valid_neg'],
        'test_neg': data['test_neg']
    }
    
    for split_name, split_edges in splits.items():
        edges = split_edges.cpu().numpy()
        results = {}
        
        if use_len_spd:
            results['len_sp'] = []
        if use_num_spd:
            results['num_sp'] = []
        if use_cnb_jac:
            results['jac'] = []
        if use_cnb_aa:
            results['aa'] = []
        if use_cnb_ra:
            results['ra'] = []
        
        for (u, v) in edges:
            try:
                if use_len_spd:
                    results['len_sp'].append(nx.shortest_path_length(G, u, v))
                
                if use_num_spd or use_cnb_jac or use_cnb_aa or use_cnb_ra:
                    spaths = list(nx.all_shortest_paths(G, u, v))
                    
                    if use_num_spd:
                        results['num_sp'].append(len(spaths))
                    
                    if use_cnb_jac or use_cnb_aa or use_cnb_ra:
                        neighbors_u = set(G.neighbors(u))
                        neighbors_v = set(G.neighbors(v))
                        
                        if use_cnb_jac:
                            jac = len(neighbors_u & neighbors_v) / len(neighbors_u | neighbors_v)
                            results['jac'].append(jac if not np.isnan(jac) else 0)
                            
                        if use_cnb_aa or use_cnb_ra:
                            common = neighbors_u & neighbors_v
                            if use_cnb_aa:
                                aa = sum(1/np.log(G.degree(w)) for w in common if G.degree(w) > 1)
                                results['aa'].append(aa if not np.isnan(aa) else 0)
                            if use_cnb_ra:
                                ra = sum(1/G.degree(w) for w in common)
                                results['ra'].append(ra if not np.isnan(ra) else 0)
            
            except nx.NetworkXNoPath:
                # 处理不可达的情况
                default_val = 0 if split_name.endswith('neg') else np.nan
                if use_len_spd: results['len_sp'].append(-1)
                if use_num_spd: results['num_sp'].append(0)
                if use_cnb_jac: results['jac'].append(0)
                if use_cnb_aa: results['aa'].append(0)
                if use_cnb_ra: results['ra'].append(0)
        
        # 将结果转换为Tensor
        for metric in results:
            tensor = torch.FloatTensor(results[metric])
            if split_name.endswith('neg') and metric == 'len_sp':
                tensor = torch.abs(tensor)  # 负样本处理为绝对值
            heuristics[f'{split_name}_{metric}'] = tensor.to(data['edge_index'].device)
    
    # 合并到原始数据
    data.update(heuristics)
    print("Computing heuristics done!")
    return data

def compute_all_pairs_heuristics(data, use_len_spd=True, use_num_spd=False,
                                use_cnb_jac=True, use_cnb_aa=True, use_cnb_ra=True,
                                cache_dir=None, force_recompute=False):
    print("Computing all-pairs heuristics...")
    num_nodes = data['num_nodes']
    edge_index = data['edge_index']
    device = data['edge_index'].device

    if cache_dir is None:
        cache_dir = os.path.join(DATA_DIR, data['dataset'], "heuristics_cache")
    
    # Generate unique cache key based on graph structure and parameters
    # edge_hash = hashlib.md5(data['dataset'].tobytes()).hexdigest()
    params = f"{use_len_spd}{use_num_spd}{use_cnb_jac}{use_cnb_aa}{use_cnb_ra}"
    cache_file = os.path.join(cache_dir, f"{params}.pt")
    
    if not force_recompute and os.path.exists(cache_file):
        print("Loading cached heuristics...")
        heuristics = torch.load(cache_file)
        data.update(heuristics)
        return data
    
    start = time()

    # Build undirected graph
    G = to_networkx(
        Data(
            x=data['x'],
            edge_index=edge_index
        ),
        to_undirected=True
    )
    len_shortest_path = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    num_shortest_path = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    undir_jac = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    undir_aa = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    undir_ra = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    for i in range(1, num_nodes):
        for j in range(i+1, num_nodes):
            s_idx = i
            o_idx = j
            try:
                if use_len_spd:
                    len_shortest_path[s_idx][o_idx] = nx.shortest_path_length(G, s_idx, o_idx)
                    len_shortest_path[o_idx][s_idx] = nx.shortest_path_length(G, s_idx, o_idx)
                if use_num_spd:
                    shortest_path_list = [p for p in nx.all_shortest_paths(G, s_idx, o_idx)]
                    num_shortest_path[s_idx][o_idx] = len(shortest_path_list)
                    num_shortest_path[o_idx][s_idx] = len(shortest_path_list)
                if use_cnb_jac:
                    preds = nx.jaccard_coefficient(G, [(s_idx, o_idx)])
                    _, _, jac = next(preds)
                    undir_jac[s_idx][o_idx] = jac
                    undir_jac[o_idx][s_idx] = jac
                if use_cnb_aa:
                    preds = nx.adamic_adar_index(G, [(s_idx, o_idx)])
                    _, _, aa = next(preds)
                    undir_aa[s_idx][o_idx] = aa
                    undir_aa[o_idx][s_idx] = aa
                if use_cnb_ra:
                    preds = nx.resource_allocation_index(G, [(s_idx, o_idx)])
                    _, _, ra = next(preds)
                    undir_ra[s_idx][o_idx] = ra
                    undir_ra[o_idx][s_idx] = ra
            except nx.exception.NetworkXNoPath:
                # No way between these two points
                len_shortest_path[s_idx][o_idx] = np.iinfo('long').max
                len_shortest_path[o_idx][s_idx] = np.iinfo('long').max
                shortest_path_list = []
                num_shortest_path[s_idx][o_idx] = 0
                num_shortest_path[o_idx][s_idx] = 0
            len_shortest_path.fill_diagonal_(0)
            num_shortest_path.fill_diagonal_(0)
            undir_jac.fill_diagonal_(0)
            undir_aa.fill_diagonal_(0)
            undir_ra.fill_diagonal_(0)
    len_shortest_path = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    heuristics = {len_shortest_path, num_shortest_path, undir_jac, undir_aa, undir_ra}

    # Save to cache and update data
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(heuristics, cache_file)
    for k, v in heuristics.items():
        heuristics[k] = v.to(data['edge_index'].device)
    data.update(heuristics)
    print("All-pairs heuristics computed!")
    print(f"Time: {time()-start:.2f} seconds")
    data['len_shortest_path'] = len_shortest_path.to(device)
    data['num_shortest_path'] = num_shortest_path.to(device)
    data['undir_jac'] = undir_jac.to(device)
    data['undir_aa'] = undir_aa.to(device)
    data['undir_ra'] = undir_ra.to(device)
    return data