import os
import torch
from tqdm import tqdm
from datetime import datetime   
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from util.utils import *
from train.testing import *

from models.other_models import mlp_score
from models.link_transformer import LinkTransformer
from models.SIEGphormer import SIEGphormer


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")



def train_epoch(model, score_func, data, optimizer, args, device, global_logger):
    model.train()
    score_func.train()
    train_pos = data['train_pos'].to(device)

    adjmask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)
    adjt_mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)

    total_loss = total_examples = 0
    d = DataLoader(range(train_pos.size(0)), args.batch_size, shuffle=True)
    d = tqdm(d, "Epoch")
    
    for perm in d:
        # 论文注释：这里的一个 perm 应该就是一个 batch 中，正样本所对应的下标索引构成的张量。
        # 论文注释：这里的 edges 是啥？
        edges = train_pos[perm].t()

        # Remove positive samples from adj_mask used in calculating pairwise info
        # Only needed for positive bec. otherwise don't exist
        adjmask[perm] = 0
        # 论文注释：从 train_pos 中选择那些在 adjmask 中仍然为 1 的边（即未被移除的边）。edge2keep 将包含所有需要保留的边。
        edge2keep = train_pos[adjmask, :]
        # 论文注释：masked_adj 是稀疏邻接矩阵，用于记录忽略了样本后，有哪些节点是相连的
        masked_adj = SparseTensor.from_edge_index(edge2keep.t(), sparse_sizes=(data['num_nodes'], data['num_nodes'])).to_device(device)
        masked_adj = masked_adj.to_symmetric()
        masked_adj = masked_adj.to_torch_sparse_coo_tensor().coalesce().bool().int()
        
        # 论文注释：将 adjmask 中之前被设置为 0 的位置恢复为 1，以便在处理下一个批次时可以使用
        adjmask[perm] = 1  # For next batch + negatives

        # 论文注释：为什么作者这里需要额外搞一套？
        if args.mask_input:
            adjt_mask[perm] = 0
            edge2keep = train_pos[adjt_mask, :]
            
            masked_adjt = SparseTensor.from_edge_index(edge2keep.t(), sparse_sizes=(data['num_nodes'], data['num_nodes'])).to_device(device)
            masked_adjt = masked_adjt.to_symmetric()
            
            # For next batch
            adjt_mask[perm] = 1
        else:
            masked_adjt = None
        h = model(edges, adj_prop=masked_adjt, adj_mask=masked_adj)
        pos_out = score_func(h)
        pos_loss = -torch.log(pos_out + 1e-6).mean()

        # Just do some trivial random sampling for negative samples
        neg_edges = torch.randint(0, data['num_nodes'], (edges.size(0), edges.size(1) * args.num_negative), dtype=torch.long, device=h.device)
        h = model(neg_edges)
        neg_out = score_func(h)
        neg_loss = -torch.log(1 - neg_out + 1e-6).mean()
        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        # print("Before:", torch.softmax(model.alpha, dim=0).data)
        optimizer.step()
        optimizer.zero_grad()
        # print("After:", torch.softmax(model.alpha, dim=0).data)
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples   

    return total_loss / total_examples



def train_loop(args, train_args, data, device, loggers, seed, model_save_name, verbose, global_logger):
    """
    Train over N epochs
    """
    k_list = [20, 50, 100]
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2') if 'MRR' in loggers else None
    # 在这里切换模型 LinkTransformer or SIEGphormer
    model = SIEGphormer(train_args, data, global_logger, device=device).to(device)
    score_func = mlp_score(model.out_dim, model.out_dim, 1, args.pred_layers, train_args['pred_dropout']).to(device)
                        
    optimizer = torch.optim.Adam(list(model.parameters()) + list(score_func.parameters()), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: train_args['decay'] ** e)
    
    kill_cnt = 0
    best_valid = 0

    for epoch in range(1, 1 + args.epochs):
        print("Alpha:", torch.softmax(model.alpha, dim=0).data)
        print(f">>> Epoch {epoch} - {datetime.now().strftime('%H:%M:%S')}")

        loss = train_epoch(model, score_func, data, optimizer, args, device, global_logger)
        # global_logger.write_down(f"Epoch {epoch} Loss: {loss:.4f}")   
        if epoch % args.eval_steps == 0:
            # global_logger.write_down("Evaluating model...")
            
            if "citation" not in args.data_name.lower() or args.heart:
                results_rank = test(model, score_func, data, evaluator_hit, evaluator_mrr, args.test_batch_size, k_list, heart=args.heart)
            else:
                results_rank = test_citation2(model, score_func, data, evaluator_hit, evaluator_mrr, args.test_batch_size)

            global_logger.write_down(f"Epoch {epoch} Results:    ")
            for key, result in results_rank.items():
                loggers[key].add_result(seed, result)
                if args.metric == key:
                    global_logger.write_down(f"  {key} = {result}")
                if args.data_name == "cora":
                    if epoch in {5, 10, 25, 50, 100}:
                        global_logger.write_down(f"---------------------  {key} = {result}---------------------")

            best_valid_current = torch.tensor(loggers[eval_metric].results[seed])[:, 1].max()

            if best_valid_current > best_valid:
                kill_cnt = 0
                best_valid = best_valid_current
                if model_save_name is not None:
                    save_model(model, score_func, optimizer, model_save_name + ".pt")
            else:
                kill_cnt += 1
                
                if kill_cnt > args.kill_cnt: 
                    if True:
                        global_logger.write_down("Early Stop!")
                        global_logger.write_down(f"best_valid = {best_valid}")
                        print(model.alpha)
                    break
                    
        scheduler.step()
    
    return best_valid


def train_data(args, train_args, data, device, global_logger, verbose=True):
    """
    Run over n random seeds
    """
    init_seed(args.seed)

    if args.save_as is not None:
        model_save_name = os.path.join("checkpoints", args.data_name, args.save_as)
    else:
        model_save_name = None

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
    }
    if "citation" in data['dataset'] or data['dataset'] in ['cora', 'citeseer', 'pubmed',  'chameleon', 'squirrel'] or args.heart:
        loggers['MRR'] = Logger(args.runs)

    # Over N splits
    best_valid_results = []

    for seed in tqdm(range(0, args.runs), f"Training over {args.runs} seeds"):
        if args.runs > 1:
            init_seed(seed)

        run_save_name = model_save_name
        if model_save_name is not None and args.runs > 1:
            run_save_name = model_save_name + f"_seed-{seed+1}"

        best_valid = train_loop(args, train_args, data, device, loggers, seed, run_save_name, verbose, global_logger)
        best_valid_results.append(best_valid)

    for key in loggers.keys():     
        if key == args.metric:
            global_logger.write_down(key + "\n" + "-" * len(key))  
            # Both lists. [0] = Train, [1] = Valid, [2] = Test
            best_mean, best_var = loggers[key].print_statistics()
    global_logger.write_down(f'Highest Valid: {best_mean[1]:.2f} ± {best_valid[1]:.2f}')
    global_logger.write_down(f'Final Test: {best_mean[2]:.2f} ± {best_valid[2]:.2f}')
    return best_mean[1], f"{best_mean[1]} ± {best_var[1]}", f"{best_mean[2]} ± {best_var[2]}"




