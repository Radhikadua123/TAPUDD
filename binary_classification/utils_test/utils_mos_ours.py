import os
import cv2
import torch
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torchvision.transforms as transforms
CUDA_LAUNCH_BLOCKING=1

from torch.utils.data import Dataset
import argparse
import torchvision as tv
import torch
import numpy as np
import sklearn.metrics as sk
from utils_test.mahalanobis_ours import *
from utils_test import log
from utils_test.test_utils import arg_parser

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
from utils import *
from dataset import *

    
def mk_id_ood(ood_feats, in_feats):
    """Returns train and validation datasets."""
    in_set = Dataset_ood_test(in_feats)
    ood_set = Dataset_ood_test(ood_feats)
    print(f"Using an in-distribution set with {len(in_set)} images.")
    print(f"Using an out-of-distribution set with {len(ood_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False)
    out_loader = torch.utils.data.DataLoader(
        ood_set, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False)
    
    return in_loader, out_loader


class DatasetWithMetaGroup(Dataset):
    def __init__(self, penul_feats, clusters, num_group=8):
        super(DatasetWithMetaGroup, self).__init__()
        self.penul_feats = penul_feats
        self.clusters = clusters
        self.num_group = num_group

        self.num = len(self.penul_feats)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
            
        feats = self.penul_feats[idx]
        cluster = self.clusters[idx]
        labels = np.zeros(self.num_group, dtype=np.int)
        labels[cluster] =  1
        return feats, labels


class Dataset_ood_test(Dataset):
    def __init__(self, penul_feats):
        super(Dataset_ood_test, self).__init__()
        self.penul_feats = penul_feats
        self.num = len(self.penul_feats)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
            
        feats = self.penul_feats[idx]

        return feats
        

def data_cluster_classification(train_Df, num_of_clusters):
    ####### Train data from all clusters for cluster classification #######
    df = list()
    for i in range(num_of_clusters):
        l = len(train_Df.groupby(['Cluster']).get_group(i))
        p = 0
        q = int(60*l/100)
        df.append(train_Df.groupby(['Cluster']).get_group(i)[p:q])

    # Combine images from positive and negative class(males and females) and shuffle the data.
    frames = [df[i] for i in range(num_of_clusters)]
    df3 = pd.concat(frames)
    df_train = df3.sample(frac = 1)


    ####### Val data from all clusters for cluster classification #######
    df = list()
    for i in range(num_of_clusters):
        l = len(train_Df.groupby(['Cluster']).get_group(i))
        p = int(60*l/100) + 1
        q = int(80*l/100)
        df.append(train_Df.groupby(['Cluster']).get_group(i)[p:q])

    # Combine images from positive and negative class(males and females) and shuffle the data.
    frames = [df[i] for i in range(num_of_clusters)]
    df3 = pd.concat(frames)
    df_val = df3.sample(frac = 1)


    ####### test data from all clusters for cluster classification #######
    df = list()
    for i in range(num_of_clusters):
        l = len(train_Df.groupby(['Cluster']).get_group(i))
        p = int(80*l/100) + 1
        q = l
        df.append(train_Df.groupby(['Cluster']).get_group(i)[p:q])

    # Combine images from positive and negative class(males and females) and shuffle the data.
    frames = [df[i] for i in range(num_of_clusters)]
    df3 = pd.concat(frames)
    df_test = df3.sample(frac = 1)


    ############ penultimate features and cluster label needed for cluster classification #############
    train_feats = df_train["feats"].tolist()
    train_clusters = df_train["Cluster"].tolist()

    val_feats = df_val["feats"].tolist()
    val_clusters = df_val["Cluster"].tolist()

    test_feats = df_test["feats"].tolist()
    test_clusters = df_test["Cluster"].tolist()
    
    num_groups = num_of_clusters
    train_set = DatasetWithMetaGroup(train_feats, train_clusters, num_group=num_groups)
    val_set = DatasetWithMetaGroup(val_feats, val_clusters, num_group=num_groups)
    test_set = DatasetWithMetaGroup(test_feats, test_clusters, num_group=num_groups)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


def get_group_slices(num_groups):
    group_slices = []
    start = 0
    for i in range(num_groups):
        """[1 for others and 1 for cluster]"""
        end = start + 1 + 1  
        group_slices.append([start, end])
        start = end
    return torch.LongTensor(group_slices)

def calc_group_softmax_loss(criterion, logits, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]
        group_label = labels[:, i]
        loss += criterion(group_logit, group_label)
    return loss

def calc_group_softmax_acc(logits, labels, group_slices, device):
    num_groups = group_slices.shape[0]
    loss = 0
    num_samples = logits.shape[0]

    all_group_max_score, all_group_max_class = [], []

    smax = torch.nn.Softmax(dim=-1).to(device)
    cri = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]
        group_label = labels[:, i]
        loss += cri(group_logit, group_label)

        group_softmax = smax(group_logit)
        group_softmax = group_softmax[:, 1:]    # disregard others category
        group_max_score, group_max_class = torch.max(group_softmax, dim=1)
        group_max_class += 1     # shift the class index by 1

        all_group_max_score.append(group_max_score)
        all_group_max_class.append(group_max_class)

    all_group_max_score = torch.stack(all_group_max_score, dim=1)
    all_group_max_class = torch.stack(all_group_max_class, dim=1)

    final_max_score, max_group = torch.max(all_group_max_score, dim=1)

    pred_cls_within_group = all_group_max_class[torch.arange(num_samples), max_group]

    gt_class, gt_group = torch.max(labels, dim=1)

    selected_groups = (max_group == gt_group)

    pred_acc = torch.zeros(logits.shape[0]).bool().to(device)

    pred_acc[selected_groups] = (pred_cls_within_group[selected_groups] == gt_class[selected_groups])

    return loss, pred_acc

def define_model_cluster_classification(device, num_logits):
    """function to define the model"""
    model = nn.Sequential(
            nn.Linear(in_features = 128, out_features = num_logits, bias = True),
            )
    model = model.to(device)
    return model

def train(seed, train_loader, val_loader, test_loader, num_of_clusters, device, result_path):
    """ Function to train the model."""

    ##############################
    # define model/log pth
    ##############################
    
    model_pth = result_path
    log_pth = result_path
    os.makedirs(model_pth, exist_ok=True)
    os.makedirs(log_pth, exist_ok=True)

    best_file = os.path.join(result_path, "cluster_classifier_k_{}.pt".format(num_of_clusters))
    log_file = os.path.join(result_path, "cluster_classifier_train_log_k_{}.txt".format(num_of_clusters))


    with open(log_file, "w") as file:
        file.write("")

    num_groups = num_of_clusters
    num_logits = 2*num_groups  
    print(num_logits)
    group_slices = get_group_slices(num_groups)
    group_slices.to(device)
            

    model = define_model_cluster_classification(device, num_logits)
    model.to(device)
    cri = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ##############################
    # training
    ##############################
    iteration_for_summary = 0
    best_acc = 0
    num_groups = num_of_clusters
    n_epochs_stop = 30
    epochs_no_improve = 0
    early_stop = True
    
    for epoch in range(300):
        start_time = time.process_time()
        model.train()
        running_loss = 0.0
        
        total = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            ########################
            # calc total loss
            ########################
            optimizer.zero_grad()
            logits = model(x)
            loss = calc_group_softmax_loss(cri, logits, y, group_slices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
        total += y.size(0)
        table = '[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss / total)

        with open(log_file, "a") as file:
            file.write(table)

        #######################
        # validation
        #######################
        with torch.no_grad():
            model.eval()
            all_c, all_top1 = [], []
            for data in val_loader:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                c, top1 = calc_group_softmax_acc(logits, y, group_slices, device)
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
            
            
        val_loss = np.mean(all_c)
        val_acc = np.mean(all_top1)
        table = 'Epoch: {}, Validation acc: {}, epoch time : {} seconds'.format(epoch+1, val_acc,  time.process_time() - start_time)
        if(epoch == 0):
            best_acc = val_acc
            torch.save(model.state_dict(), best_file)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_file)         
            table+= "   <<< best acc"
        else:
            epochs_no_improve += 1

        if epoch > n_epochs_stop and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            print(table)
            with open(log_file, "a") as file:
                file.write(table)
                file.write("\n")
            break

        print(table)
        with open(log_file, "a") as file:
            file.write(table)
            file.write("\n")


def test(seed, test_loader, num_of_clusters, device, result_path): 
    """ Function to test classification performance of the trained model on in-distribution test data. Calculates accuracy on in-distribution test data."""       
    log_pth = result_path
    os.makedirs(log_pth, exist_ok=True)
    log_file = os.path.join(result_path, "cluster_classifier_test_log_k_{}.txt".format(num_of_clusters))
    with open(log_file, "w") as file:
        file.write("")
    
    num_groups = num_of_clusters
    num_logits = 2*num_groups  
    print(num_logits)
    group_slices = get_group_slices(num_groups)
    group_slices.to(device)

    model = define_model_cluster_classification(device, num_logits)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(result_path, "cluster_classifier_k_{}.pt".format(num_of_clusters))))
    cri = torch.nn.CrossEntropyLoss().to(device)
    
    model.eval()

    with torch.no_grad():
        model.eval()
        all_c, all_top1 = [], []
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            c, top1 = calc_group_softmax_acc(logits, y, group_slices, device)
            all_c.extend(c.cpu())  # Also ensures a sync point.
            all_top1.extend(top1.cpu())
            
            
    test_loss = np.mean(all_c)
    test_acc = np.mean(all_top1)
    print('Test loss: {}, test acc: {}'.format(test_loss, test_acc))
    table = 'Test acc: {}'.format(test_acc)
    with open(log_file, "a") as file:
        file.write(table)


def cal_ood_score(logits, group_slices, device):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []

    smax = torch.nn.Softmax(dim=-1).to(device)
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def iterate_data(data_loader, model, group_slices, device):
    confs_mos = []
    for b, x in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            logits = model(x)
            conf_mos = cal_ood_score(logits, group_slices, device)
            confs_mos.extend(conf_mos)

    return np.array(confs_mos)

