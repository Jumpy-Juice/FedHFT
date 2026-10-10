from algorithm.Base import BaseServer, BaseClient
import os
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import copy  
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse,coalesce,degree
from utils.utils import *



def compute_pagerank_importance(edge_index, num_nodes, alpha=0.85, iters=10):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    d = degree(edge_index[0], num_nodes).view(-1, 1)
    d_inv = 1.0 / d
    d_inv[torch.isinf(d_inv)] = 0
    M = adj * d_inv
    v = torch.ones((num_nodes, 1), device=edge_index.device) / num_nodes
    for _ in range(iters):
        v = alpha * torch.mm(M, v) + (1 - alpha) / num_nodes
    return v.view(-1)

class FedHFTServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedHFTServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes= args.num_classes
        self.noise_dim = args.noise_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=args.weight_decay)
        self.lambda_r = args.lambda_r
        self.lambda_n = args.lambda_n
        self.M = args.M
        self.T_G = args.T_G

        
        self.personalized_models = {cid: copy.deepcopy(model).to(self.device) for cid in range(len(self.clients))}
        
        self.p_optimizers = {
            cid: torch.optim.Adam(self.personalized_models[cid].parameters(), 
                                  lr=0.01, 
                                  weight_decay=args.weight_decay) 
            for cid in range(len(self.clients))
        }
        
         
         
    def aggregate(self):
        print('---------------------------')
        print('knowledge integration...')
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])

        
        client_prototypes = {cid: self.clients[cid].output_prototype() for cid in self.sampled_clients}

        client_loss_vecs = {}
        client_masks = {}
        self.model.eval()
        
        y_true = torch.arange(self.num_classes, device=self.device)
        
        with torch.no_grad():
            for cid, proto in client_prototypes.items():
                mask = (proto.abs().sum(dim=1) > 0) 
                client_masks[cid] = mask
                
                temp_data = Data(x=proto, edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device))
                _, out = self.model(temp_data)
                
                loss_vec = F.nll_loss(out, y_true, reduction='none') 
                client_loss_vecs[cid] = loss_vec
        
        cids = list(self.sampled_clients)
        num_c = len(cids)
        sim_matrix = torch.zeros((num_c, num_c), device=self.device)
        
        for i in range(num_c):
            for j in range(num_c):
                m_i = client_masks[cids[i]]
                m_j = client_masks[cids[j]]
                
                m_inter = m_i & m_j
                m_union = m_i | m_j
                
                if m_inter.sum() == 0:
                    sim_matrix[i, j] = 0.0
                else:
                    vec_i = client_loss_vecs[cids[i]][m_inter]
                    vec_j = client_loss_vecs[cids[j]][m_inter]
                    cos_sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).squeeze()
                    
                    jaccard = m_inter.float().sum() / m_union.float().sum()
                    sim_matrix[i, j] = cos_sim * jaccard
        
        weights_matrix = F.softmax(sim_matrix / self.args.tau, dim=1)

        print('---------------------------')
        print('global model training...')
        for i, cid in enumerate(self.sampled_clients):

            num_gen = int(self.clients[cid].data.x.shape[0])
            generator = self.clients[cid].generator

            for _ in range(self.M):
                with torch.no_grad():
                    z_noise = torch.randn((num_gen, self.noise_dim), device=self.device).float()
                    y_noise = gen_y_noise(num_gen,self.num_classes,self.device)
                    node_logits = generator(z_noise,y_noise)
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=5).to(self.device)

                
                
                for _ in range(self.T_G):
                    for target_idx, target_cid in enumerate(cids):
                        p_model = self.personalized_models[target_cid]
                        weight = weights_matrix[target_idx, i] 
                        if weight < 0.01: continue 
                        
                       
                        p_optimizer = self.p_optimizers[target_cid]
                        
                        p_model.train()
                        p_optimizer.zero_grad()
                        _, out_p = p_model(pseudo_graph)
                        
                        
                        loss_p = weight * F.nll_loss(out_p, y_noise)
                        loss_p.backward()
                        p_optimizer.step()

        
        for cid in range(len(self.clients)):
            self.clients[cid].model.load_state_dict(self.personalized_models[cid].state_dict())




class FedHFTClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedHFTClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes= args.num_classes
        self.noise_dim = args.noise_dim
        self.generator = Generator(noise_dim=self.noise_dim,
                                   input_dim=self.feature_dim,
                                   output_dim=self.num_classes,
                                   dropout=0.2).to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lambda_d = args.lambda_d
        self.lambda_f = args.lambda_f
        
        self.Lambda_fis = getattr(args, 'Lambda_fis', 0.05)
        
       
        self.Lambda_sem = getattr(args, 'Lambda_sem', 0.0)

        self.topk = getattr(args, 'topk', 0.5)
        

        self.seed = int(getattr(args, 'seed', 0))
        self._np_rng = np.random.default_rng(self.seed)
        self.last_times = {}

        
        self._proto = torch.zeros((self.num_classes, self.feature_dim), device=self.device)
        

    def _sample_strict_non_neighbors(self, edge_index: Tensor, center_nodes: Tensor, num_nodes: int, max_tries: int = 20) -> Tensor:

        if center_nodes.numel() == 0:
            return center_nodes

        ei = edge_index.detach().to('cpu')
        row = ei[0].tolist()
        col = ei[1].tolist()
        neighbors = [set() for _ in range(num_nodes)]
        for r, c in zip(row, col):
            if r == c:
                continue
            neighbors[r].add(c)
            neighbors[c].add(r)

        centers_cpu = center_nodes.detach().to('cpu').tolist()
        neg_list = []
        for u in centers_cpu:
            v = int(self._np_rng.integers(0, num_nodes))
            for _ in range(max_tries):
                v = int(self._np_rng.integers(0, num_nodes))
                if v != u and v not in neighbors[u]:
                    break
            if v == u:
                v = (u + 1) % num_nodes
            neg_list.append(v)

        return torch.tensor(neg_list, device=self.device, dtype=torch.long)


    def _update_proto_from_real(self):
       
        if torch.any(self._proto != 0):
            return
            
        num_nodes = self.data.x.shape[0]
        
        
        pr_weights = compute_pagerank_importance(self.data.edge_index, num_nodes).to(self.device)

        with torch.no_grad():
            x = self.data.x.to(self.device)
            y = self.data.y.to(self.device)
            train_mask = self.data.train_mask.to(self.device)
            
            for c in range(self.num_classes):
                idx = train_mask & (y == c)
                if idx.any():
                   
                    c_weights = pr_weights[idx]
                    c_features = x[idx]
                    
                   
                    weighted = (c_features * c_weights.view(-1, 1)).sum(dim=0) / (c_weights.sum() + 1e-8)

                    
                    self._proto[c] = weighted

    def _gdn_fl_loss(self, node_logits, grads_idx):
        
        train_mask = self.data.train_mask.to(self.device)
        labels = self.data.y.to(self.device)

        x_log = F.log_softmax(node_logits[train_mask][:, grads_idx], dim=-1)
        pos_labels = labels[train_mask]
        pos_proto = self._proto[pos_labels][:, grads_idx]

        
        if self.num_classes > 1:
            offset = torch.randint(1, self.num_classes, (pos_labels.size(0),), device=self.device)
            neg_labels = (pos_labels + offset) % self.num_classes
            neg_proto = self._proto[neg_labels][:, grads_idx]
        else:
            neg_proto = pos_proto.detach()

        target_pos = F.softmax(pos_proto, dim=-1)
        target_neg = F.softmax(neg_proto, dim=-1)

        loss_pos = F.kl_div(x_log, target_pos, reduction='batchmean')
        loss_neg = F.kl_div(x_log, target_neg, reduction='batchmean')
        return loss_pos, loss_neg


    def fish_calculate(self, net, loader):
        average_grads = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
        num_batches = 0

        for step, (image, label) in enumerate(loader):
            num_batches += 1

           
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)
            if hasattr(label, 'to'):
                label = label.to(self.device)

            if isinstance(image, Data):
                result = net(image)
            else:
                temp_data = Data(x=image, edge_index=self.data.edge_index.to(self.device))
                result = net(temp_data)

            if isinstance(result, (tuple, list)):
                output = result[1]
            else:
                output = result

            if hasattr(self.data, 'train_mask'):
                output = output[self.data.train_mask]
                label = label[self.data.train_mask]

            pre = F.log_softmax(output, dim=1)
            log_likelihoods = (pre[:, label])

            net.zero_grad()
            log_likelihoods.mean().backward()

            for name, param in net.named_parameters():
                if param.grad is not None:
                    average_grads[name] += param.grad

            net.zero_grad()

        for name in average_grads:
            average_grads[name] /= num_batches

        fish_dict = {}
        for name, grad in average_grads.items():
            fish_dict[name] = average_grads[name].clone() ** 2

        return fish_dict
    
    def output_prototype(self):
        self.generator.eval()
        num_nodes = self.data.x.shape[0]
      
        pr_weights = compute_pagerank_importance(self.data.edge_index, num_nodes)
        
        with torch.no_grad():
            z = torch.randn((num_nodes, self.noise_dim), device=self.device)
            
            node_features = self.generator(z, self.data.y)
            
            prototypes = []
            for c in range(self.num_classes):
                mask = (self.data.y == c)
                if mask.sum() > 0:
                    c_weights = pr_weights[mask]
                    c_features = node_features[mask]
                    
                    weighted_proto = (c_features * c_weights.view(-1, 1)).sum(dim=0) / (c_weights.sum() + 1e-8)
                    prototypes.append(weighted_proto)
                else:
                    prototypes.append(torch.zeros(self.feature_dim, device=self.device))
        
        return torch.stack(prototypes)   


    def train(self):
        self.optimizer.zero_grad()
        num_gen = self.data.x.shape[0]

      
        real_loader = [(self.data, self.data.y)]

        
        target_fish = self.fish_calculate(self.model, real_loader)


        self.z_noise = torch.randn((num_gen, self.noise_dim), device=self.device).float()

        node_logits = self.generator(self.z_noise,self.data.y)
        node_norm = F.normalize(node_logits, p=2, dim=1)
        adj_logits = torch.mm(node_norm, node_norm.t())
        pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=5)
        self.model.eval()
        self.model.zero_grad()

        
        result_pseudo = self.model(pseudo_graph)
        if isinstance(result_pseudo, (tuple, list)):
            out_pseudo = result_pseudo[1]
        else:
            out_pseudo = result_pseudo

        
        if hasattr(self.data, 'train_mask'):
            out_pseudo = out_pseudo[self.data.train_mask]
            y_pseudo = self.data.y[self.data.train_mask]
        else:
            y_pseudo = self.data.y

        
        pre_pseudo = F.log_softmax(out_pseudo, dim=1)

        log_likelihoods_pseudo = pre_pseudo.gather(1, y_pseudo.view(-1, 1)).squeeze()

        
        grads_pseudo = torch.autograd.grad(
            log_likelihoods_pseudo.mean(),
            self.model.parameters(),
            create_graph=True
        )

        
        fisher_start = time.perf_counter()
        loss_fisher = 0
        for (name, _), grad_p in zip(self.model.named_parameters(), grads_pseudo):
            if name in target_fish:
                fish_val_pseudo = grad_p ** 2
                fish_val_real = target_fish[name].detach()

                loss_fisher += torch.sum((fish_val_pseudo - fish_val_real) ** 2)
        fisher_time = time.perf_counter() - fisher_start

        sem_time = 0.0
        loss_sem = torch.tensor(0.0, device=self.device)
        if float(self.Lambda_sem) > 0.0:
            sem_start = time.perf_counter()
            self._update_proto_from_real()

            self.model.eval()
            self.model.zero_grad()
            temp_x = node_logits.detach().requires_grad_(True)
            temp_graph = Data(x=temp_x, edge_index=pseudo_graph.edge_index.to(self.device))
            _, out = self.model(temp_graph)
            probe_loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            grad = torch.abs(torch.autograd.grad(outputs=probe_loss, inputs=temp_x)[0])

            k = max(1, int(self.feature_dim * 0.5))
            if k > 0:
                k = min(k, self.feature_dim)
                grads_idx = grad.mean(dim=0).topk(k=k).indices
                
                if float(self.Lambda_sem) > 0.0:
                    loss_pos, loss_neg = self._gdn_fl_loss(node_logits.to(self.device), grads_idx)
                    loss_sem = float(self.Lambda_sem) * torch.exp(loss_pos - loss_neg)
            sem_time = time.perf_counter() - sem_start
      


        
        graph_start = time.perf_counter()
        A_ori = to_dense_adj(edge_index=self.data.edge_index,max_num_nodes=num_gen).squeeze(0)
        A_new = to_dense_adj(edge_index=pseudo_graph.edge_index,max_num_nodes=num_gen).squeeze(0)
        A_new = A_new // 2
        graph_time = time.perf_counter() - graph_start
        
        feat_start = time.perf_counter()
        loss_feat = feature_loss(pseudo_graph.x[self.data.train_mask],self.data.x[self.data.train_mask],lambda_d = self.lambda_d)
        feat_time = time.perf_counter() - feat_start
        
        fgw_start = time.perf_counter()
        loss_fgw = gw_distance_loss(A_ori[self.data.train_mask][:, self.data.train_mask],
                                A_new[self.data.train_mask][:, self.data.train_mask],
                                self.data.x[self.data.train_mask],
                                pseudo_graph.x[self.data.train_mask],self.device)
        fgw_time = time.perf_counter() - fgw_start
        
        loss = loss_feat + self.lambda_f * loss_fgw + self.Lambda_fis * loss_fisher + loss_sem
        
        loss.backward()
        self.optimizer.step()
        self.last_times = {
            "fedhft_fisher_time_sec": fisher_time,
            "fedhft_sem_time_sec": sem_time,
            "fedhft_graph_time_sec": graph_time,
            "fedhft_feat_loss_time_sec": feat_time,
            "fedhft_fgw_time_sec": fgw_time,
        }
        return loss.item()
