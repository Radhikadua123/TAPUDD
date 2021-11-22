import torch.nn as nn
import torch
import numpy as np
import random
import os


def get_model_pth(args):
    model_dir =os.path.join(args.root_path, args.loss, args.data, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model_{}.pt".format(args.seed))
    return model_path

def get_log_pth(args):
    log_dir = os.path.join(args.root_path, args.loss, args.data, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test_log_{}.txt".format(args.seed))
    return log_path

def get_result_pth(args):
    result_dir = os.path.join(args.root_path, args.loss, args.data, 'scores', args.metric)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def define_model(input_size = 2, hidden_size=16, num_classes= 2):
    model = Vanilla(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    return model

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    
class Vanilla(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Vanilla, self).__init__()
        net = [nn.Linear(input_size, hidden_size) ,
                nn.ReLU(),]
        for i in range(7):
            net += [nn.Linear(hidden_size, hidden_size) ,
                nn.ReLU(),]
        # net += [nn.Linear(hidden_size, num_classes)]

        self.net = nn.Sequential(*net)
        self.logit = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        penulti = self.net(x)
        out = self.logit(penulti)

        return out, penulti


    


