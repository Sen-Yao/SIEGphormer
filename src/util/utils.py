import os
import sys
import math
import json
import torch
import random
import logging
import numpy as np

import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from datetime import datetime


def init_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def save_model(model, score_func, optimizer, save_path):
    """
    Save model
    """
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    print(f"Saving model to {save_path}...")

    state = {
        'model'	: model.state_dict(),
        "score_func": score_func.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, save_path)


def load_model(model, score_func, checkpoint, device):
    """
    Load saved models
    """
    state = torch.load(checkpoint, map_location="cpu")
    keys = model.load_state_dict(state["model"]) #, strict=False)
    # print("Model Unmatched Params", keys, flush=True)
    keys = score_func.load_state_dict(state['score_func']) #, strict=False)
    # print("Model Unmatched Params", keys, flush=True)

    model = model.to(device)
    score_func = score_func.to(device)

    return model, score_func


def get_logger(name, log_dir, config_dir):	
    """
	Creates a logger object
	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
	"""
    config_dict = json.load(open( config_dir + '/log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


class Logger(object):
    def __init__(self, runs, log_path=None, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.log_path = log_path

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)
        self.write_down(result)


    def get_best_epochs(self, eval_steps, run=None):
        # Return the epoch with the best val performance for each seed
        best_results = []

        for r in self.results:
            r = torch.tensor(r)
            best_val_epoch = eval_steps * (r[:, 1].argmax() + 1)
            best_results.append(best_val_epoch)
        
        return best_results


    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            # print(f'All runs:')

            r = best_result[:, 0]
            # print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1]
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 2]
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            # print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 3]
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]

            # return best_valid, best_valid_mean, mean_list, var_list
            return mean_list, var_list
    
    def save_args(self, cmd_args, args):
        text = "\n\nTraining arguments:\n"
        for arg_name, value in args.items():
            text = text + str(arg_name) + ": " + str(value) + "\n"
        self.write_down(text)
    def write_down(self, text):
        
        if self.log_path is not None:
            with open(self.log_path, 'a', encoding='utf-8') as file:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                text = current_time + ': ' + text
                print(text)
                file.write(text + '\n')


def rename_best_saved(logger, model_save_name, eval_steps, rand_split=False):
    """
    Rename the file name for the best performing model

    We basically just remove the epoch info
    """
    num_runs = len(logger.results)
    runtype = "split" if rand_split else "seed"

    for run, epoch in enumerate(logger.get_best_epochs(eval_steps)):
        if num_runs > 1:
            existing_name = f"{model_save_name}_{runtype}-{run+1}_epoch-{epoch}.pt"
            new_name = f"{model_save_name}_{runtype}-{run+1}.pt"
        else:
            existing_name = f"{model_save_name}_epoch-{epoch}.pt"
            new_name = f"{model_save_name}.pt"

        os.rename(existing_name, new_name)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    device = torch_sparse.device
    torch_sparse = torch_sparse.to("cpu")
    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


def drnl_node_labeling(adj, batch_src, batch_dst):
    """
    批量计算DRNL节点标签。
    
    参数:
    adj: scipy.sparse.csr_matrix，图的邻接矩阵
    batch_src: 一维Tensor，批量的源节点索引
    batch_dst: 一维Tensor，批量的目标节点索引
    
    返回:
    batch_z: (batch_size, num_nodes) 的LongTensor，每个样本的节点标签
    """
    device = batch_src.device
    num_nodes = adj.shape[0]
    batch_size = batch_src.size(0)
    batch_z = []

    
    # 遍历每个样本
    for i in range(batch_size):
        src = batch_src[i].item()
        dst = batch_dst[i].item()
        
        # 确保 src <= dst
        src, dst = (dst, src) if src > dst else (src, dst)
        
        # 构建排除src后的邻接矩阵
        idx_wo_src = list(range(src)) + list(range(src + 1, num_nodes))
        adj_wo_src = adj[idx_wo_src, :][:, idx_wo_src]
        
        # 构建排除dst后的邻接矩阵
        idx_wo_dst = list(range(dst)) + list(range(dst + 1, num_nodes))
        adj_wo_dst = adj[idx_wo_dst, :][:, idx_wo_dst]
        
        # 计算到src的最短路径（排除dst节点）
        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
        dist2src = np.insert(dist2src, dst, 0)  # 插入被排除的dst节点距离为0
        
        # 计算到dst的最短路径（排除src节点）
        # 注意：在排除src后的邻接矩阵中，原dst节点的索引为dst-1（因为src <= dst）
        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
        dist2dst = np.insert(dist2dst, src, 0)  # 插入被排除的src节点距离为0
        
        # 转换为Tensor
        dist2src = torch.from_numpy(dist2src).float().to(device)
        dist2dst = torch.from_numpy(dist2dst).float().to(device)
        
        # 计算标签
        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2
        
        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.
        
        batch_z.append(z.to(torch.long))
    
    # 堆叠结果 (batch_size, num_nodes)
    return torch.stack(batch_z, dim=0).to(torch.float32)
