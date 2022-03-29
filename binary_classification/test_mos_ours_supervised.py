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
from utils_test.test_utils import stable_cumsum, fpr_and_fdr_at_recall, get_measures, plot_aupr_auroc
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
from utils import *
from dataset import *
from utils_test.utils_mos_ours import *


def run_eval(logger, model, args, in_loader, out_loader, group_slices, device):
    # switch to evaluate mode
    model.eval()
    logger.info("Running test...")
    logger.flush()
    
    in_confs = iterate_data(in_loader, model, group_slices, device)

    out_confs = iterate_data(out_loader, model, group_slices, device)

    in_examples = in_confs.reshape((-1, 1))
    out_examples = out_confs.reshape((-1, 1))

    dir_path = os.path.join(args.logdir, args.name, args.adjust_scale)
    os.makedirs(dir_path, exist_ok=True)
    file_path_ood_scores = os.path.join(args.logdir, args.name, args.adjust_scale, "ood_scores.npy")
    dir_path = os.path.join(args.logdir, args.name, "1.0")
    os.makedirs(dir_path, exist_ok=True)
    file_path_id_scores = os.path.join(args.logdir, args.name, "1.0", "ood_scores.npy")

    np.save(file_path_id_scores, in_examples)
    np.save(file_path_ood_scores, out_examples)

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    os.makedirs(os.path.join(args.logdir, args.name, args.adjust_scale), exist_ok=True)
    path_auroc_aupr = os.path.join(args.logdir, args.name, args.adjust_scale)
    auc_roc, auc_aupr_in, auc_aupr_out = plot_aupr_auroc(in_examples, out_examples, path_auroc_aupr)

    logger.info('=================== Results for GMM + Mahala =================')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))
    logger.info('Recalculating AUROC using sk.auc: {}'.format(auc_roc))
    logger.info('Recalculating AUPR (In) using sk.auc: {}'.format(auc_aupr_in))
    logger.info('Recalculating AUPR (Out) using sk.auc: {}'.format(auc_aupr_out))

    logger.flush()


def unsupervised_ood_detection(logger, args, num_of_clusters, clustering_type, covar_type, use_gmm_stats, ood_path, id_path, device):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_feats_df = []

    train_feats = np.load(args.train_feats_path)
    train_labels = np.load(args.train_labels_path)
    print(train_feats.shape)
    train_feats_df.extend(train_feats)

    train_Df = pd.DataFrame(list(zip(train_feats_df)), columns=["feats"])
    
    print(len(train_Df))
    
    labels_list = []
    for i in train_labels:
        labels_list.append(i[0])

    train_Df["Cluster"] = labels_list

    train_loader, val_loader, test_loader = data_cluster_classification(train_Df, num_of_clusters)

    result_path = "results_ce/mos_our_ver_supervised_models/seed_" + str(args.seed)
    model_pth = os.path.join(result_path, "models_gaussian_grouping_mos")
    os.makedirs(model_pth, exist_ok=True)
    best_file = os.path.join(result_path, "models_gaussian_grouping_mos", "cluster_classifier_k_{}.pt".format(num_of_clusters))
    
    ## """ train model for cluster classification"""
    if os.path.exists(best_file):
        print("cluster_classification model already present")
    else:
        train(args.seed, train_loader, val_loader, test_loader, num_of_clusters, device, result_path)
        test(args.seed, test_loader, num_of_clusters, device, result_path)

    num_groups = num_of_clusters
    num_logits = 2*num_groups  
    print(num_logits)
    group_slices = get_group_slices(num_groups)
    group_slices.to(device)

    model = define_model_cluster_classification(device, num_logits)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(result_path, "models_gaussian_grouping_mos", "cluster_classifier_k_{}.pt".format(num_of_clusters))))

    ood_feats = []
    in_feats = []

    in_feats.extend(np.load(id_path))
    
    ood_feats.extend(np.load(ood_path))
    in_loader, out_loader = mk_id_ood(ood_feats, in_feats)
    
    print("############# NUMBER OF CLUSTERS ############", num_of_clusters)
    run_eval(logger, model, args, in_loader, out_loader, group_slices, device)

   

def main(args):
    logger = log.setup_logger(args)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info(f"Loading model ")

    clustering_type = "gaussian" # kmeans or gaussian
    covar_type = "full"
    use_gmm_stats = True
    print(type(args.num_of_clusters))
    args.num_of_clusters = int(args.num_of_clusters)
    print(type(args.num_of_clusters), args.num_of_clusters)
    unsupervised_ood_detection(logger, args, args.num_of_clusters, clustering_type, covar_type, use_gmm_stats, args.ood_feats_path, args.id_feats_path, device)


if __name__ == "__main__":
    parser = arg_parser()
    parser.add_argument('--result_path', help='id features path')
    parser.add_argument('--train_feats_path', help='train features path')
    parser.add_argument('--train_labels_path', help='train labels path')
    parser.add_argument('--ood_feats_path', help='ood features path')
    parser.add_argument('--id_feats_path', help='id features path')
    parser.add_argument("--num_of_clusters", required=True)
    parser.add_argument('--seed', default=0, type=int, help='set seed')
    parser.add_argument("--adjust_type", required=True)
    parser.add_argument("--adjust_scale", required=True)
    
    main(parser.parse_args())