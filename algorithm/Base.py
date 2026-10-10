import random
import time
import resource

import numpy as np
import torch.nn.functional as F
import torch


class BaseServer:
    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.T_L = args.T_L
        self.data = data
        self.num_total_samples = sum([client.num_samples for client in self.clients])

    @staticmethod
    def _gpu_mem_mb():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.max_memory_reserved() / 1024 / 1024
        return 0.0, 0.0

    @staticmethod
    def _cpu_mem_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def run(self):
        total_start = time.perf_counter()
        for round in range(self.num_rounds):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            round_start = time.perf_counter()
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            comm_start = time.perf_counter()
            self.communicate()
            comm_time = time.perf_counter() - comm_start

            print("cid : ", end='')
            train_time = 0.0
            for cid in self.sampled_clients:
                print('---------------------------')
                print(f'{cid} training proxy model...')
                client_start = time.perf_counter()
                for epoch in range(self.T_L):
                    self.clients[cid].train()
                train_time += time.perf_counter() - client_start
    
            agg_start = time.perf_counter()
            self.aggregate()
            agg_time = time.perf_counter() - agg_start
            eval_start = time.perf_counter()
            self.global_evaluate()
            eval_time = time.perf_counter() - eval_start

            round_time = time.perf_counter() - round_start
            gpu_alloc_mb, gpu_reserved_mb = self._gpu_mem_mb()
            cpu_mem_mb = self._cpu_mem_mb()
            print(f"round_time_sec: {round_time:.4f}")
            print(f"train_time_sec: {train_time:.4f}")
            print(f"comm_time_sec: {comm_time:.4f}")
            print(f"agg_time_sec: {agg_time:.4f}")
            print(f"eval_time_sec: {eval_time:.4f}")
            if self.clients and hasattr(self.clients[0], 'last_times'):
                summed_times = {}
                for client in self.clients:
                    if hasattr(client, 'last_times') and client.last_times:
                        for k, v in client.last_times.items():
                            summed_times[k] = summed_times.get(k, 0.0) + float(v)
                for k, v in summed_times.items():
                    print(f"{k}: {v:.4f}")
            print(f"gpu_max_allocated_mb: {gpu_alloc_mb:.2f}")
            print(f"gpu_max_reserved_mb: {gpu_reserved_mb:.2f}")
            print(f"cpu_rss_mb: {cpu_mem_mb:.2f}")
        total_time = time.perf_counter() - total_start
        print(f"total_time_sec: {total_time:.4f}")

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

    def sample(self):
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)
        self.sampled_clients = sampled_clients

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def global_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            _,out = self.model(self.data)
            loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
            pred = out[self.data.test_mask].max(dim=1)[1]
            acc = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            ## [NEW] 计算平均局部准确率 (Average Local Accuracy)
            local_accs = []
            for c in self.clients:
                c.model.eval()
                _, c_out = c.model(c.data)
                c_pred = c_out[c.data.test_mask].max(dim=1)[1]
                c_acc = c_pred.eq(c.data.y[c.data.test_mask]).sum().item() / c.data.test_mask.sum().item()
                local_accs.append(c_acc)
            avg_local_acc = sum(local_accs) / len(local_accs)
            
            print("test_loss : "+format(loss.item(), '.4f'))
            self.logger.write_test_loss(loss.item())
            
            print("test_acc : "+format(acc, '.4f'))
            self.logger.write_test_acc(acc)
            
            print("avg_local_acc : "+format(avg_local_acc, '.4f'))
            self.logger.write_local_acc(avg_local_acc)


class BaseClient:
    def __init__(self, args, model, data):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.data = data
        self.loss_fn = F.nll_loss
        self.num_samples = len(data.x)
        self.args = args

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        _, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
