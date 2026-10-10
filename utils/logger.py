import os


class DefaultLogger:
    def __init__(self, exp_name, log_dir):
        self.exp_name = exp_name
        self.file = open(os.path.join(log_dir, exp_name), 'w')

    def open_log_file(self, exp_name, log_dir):
        self.file = open(os.path.join(log_dir, exp_name), 'w')

    def close_log_file(self):
        self.file.close()

    def write_round(self, round):
        self.file.write("round :"+str(round)+'\n')

    def write_test_acc(self, acc):
        self.file.write("global_test_acc :"+format(acc, '.4f')+'\n')

    def write_test_loss(self, loss):
        self.file.write("test_loss :"+format(loss, '.4f')+'\n')

    def write_local_acc(self, acc):
        ## [NEW] 格式与 write_test_acc 完全对齐
        self.file.write("avg_local_acc :"+format(acc, '.4f')+'\n')
        # 确保实时写入文件，防止程序崩溃时丢失数据
        self.file.flush()    

    def write_mean_val_acc(self, mean_val_acc):
        self.file.write("client_mean_val_acc :"+format(mean_val_acc, '.4f')+'\n')

    def write_std_val_loss(self, std_val_loss):
        self.file.write("client_std_val_loss :"+format(std_val_loss, '.4f')+'\n')

    def write_mean_val_loss(self, mean_val_loss):
        self.file.write("client_mean_val_loss :"+format(mean_val_loss, '.4f')+'\n')

    def write_std_val_acc(self, std_val_acc):
        self.file.write("client_std_val_acc :"+format(std_val_acc, '.4f')+'\n')
