import os
import cv2
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.models as models
import torchvision.transforms as transforms

import sklearn.metrics as sk
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

CUDA_LAUNCH_BLOCKING=1

from utils import *
from dataset import *
from utils_test.mahalanobis_ours import *
from utils_test import log
from utils_test.test_utils import arg_parser
from utils_test.test_utils import stable_cumsum, fpr_and_fdr_at_recall, get_measures, plot_aupr_auroc

torch.set_num_threads(1)
    
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

        return feats, cluster


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



def run_eval(logger, model, args, estimator, num_groups, clustering_type, covar_type, use_gmm_stats, train_loader, in_loader, out_loader, mean, var, device, metric = 'none'):
    # switch to evaluate mode
    model.eval()
    logger.info("Running test...")
    logger.flush()
    
    model.eval()
    out_score =[]
    in_score = []
    for data in out_loader:
        data = data.to(device)
        out_confs, preds = get_Mahalanobis_score(model, data, num_classes = num_groups, sample_mean = mean, precision = var, covar_type = covar_type, layer_index = 0, device = device)
        out_score.extend(out_confs)
    for data in in_loader:
        data = data.to(device)
        in_confs, preds = get_Mahalanobis_score(model, data, num_classes = num_groups, sample_mean = mean, precision = var, covar_type = covar_type, layer_index = 0, device = device)
        in_score.extend(in_confs)
    in_examples = np.array(in_score).reshape((-1, 1))
    out_examples = np.array(out_score).reshape((-1, 1))

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



def unsupervised_ood_detection(model, logger, args, num_of_clusters, clustering_type, covar_type, use_gmm_stats, ood_path, id_path, metric, device):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_feats_df = []
    np.random.seed(1)
    train_feats = np.load(args.train_feats_path)
    train_feats_df.extend(train_feats)

    train_Df = pd.DataFrame(list(zip(train_feats_df)), columns=["feats"])
    
    print(len(train_Df))
    
    # Apply clustering
    if(clustering_type == "kmeans"):
        estimator = KMeans(n_clusters = num_of_clusters)
    else:
        estimator = GaussianMixture(n_components=num_of_clusters, covariance_type=covar_type, random_state = args.gmm_random_state)
    
    clustering_model_path = args.clustering_model_path
    os.makedirs(clustering_model_path, exist_ok=True)
    clustering_model_path_file = os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}.sav')

    if os.path.exists(clustering_model_path_file):
        estimator = pickle.load(open(clustering_model_path_file, 'rb'))
        print("Loading weights from saved clustering estimator")
        
    else:
        estimator.fit(train_feats_df)
        pickle.dump(estimator, open(clustering_model_path_file, 'wb'))
        clusters = estimator.predict(train_feats_df)
        print("Fitting estimator and saving weights of the clustering estimator")
    
    if(clustering_type == "gaussian" and covar_type == "full" and use_gmm_stats == True and os.path.exists(os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}_gmm_means.npy'))):
        mean1 = np.load(os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}_gmm_means.npy'), allow_pickle=True)
        precision = np.load(os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}_gmm_precision.npy'), allow_pickle=True)
        print("Loading mean and precision of the saved model")
        train_loader = Dataset_ood_test(train_feats) ## dummy... not used later... only added to remove error
    
    else:
        clusters = estimator.predict(train_feats_df)
        train_Df["Cluster"] = clusters
        num_groups = num_of_clusters
        num_logits = 2*num_groups 
        train_feats = train_Df["feats"].tolist()
        train_clusters = train_Df["Cluster"].tolist()

        train_set = DatasetWithMetaGroup(train_feats, train_clusters, num_group=num_groups)

        train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False)

        mean1 = estimator.means_
        precision = estimator.precisions_
    
        np.save(os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}_gmm_means.npy'), mean1)
        np.save(os.path.join(clustering_model_path, f'estimator_cluster_{num_of_clusters}_gmm_precision.npy'), precision)

        print("Calculating mean and precision of each cluster and saving them")
    mean = []
    var = []

    mean.extend([torch.from_numpy(mean1).float().to(device)])
    var.extend([torch.from_numpy(precision).float().to(device)])
    
    ood_feats = []
    in_feats = []

    in_feats.extend(np.load(id_path))
    
    ood_feats.extend(np.load(ood_path))
    in_loader, out_loader = mk_id_ood(ood_feats, in_feats)
    
    num_groups = num_of_clusters
    run_eval(logger, model, args, estimator, num_groups, clustering_type, covar_type, use_gmm_stats, train_loader, in_loader, out_loader, mean, var, device, metric)

   

def main(args):
    logger = log.setup_logger(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info(f"Loading model ")
    model = define_model()
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    model = model.cuda()
    model.eval()

    clustering_type = "gaussian" # kmeans or gaussian
    covar_type = "full"
    use_gmm_stats = True
    metric = "mahalanobis"
    print(type(args.num_of_clusters))
    args.num_of_clusters = int(args.num_of_clusters)
    print(type(args.num_of_clusters), args.num_of_clusters)
    unsupervised_ood_detection(model, logger, args, args.num_of_clusters, clustering_type, covar_type, use_gmm_stats, args.ood_feats_path, args.id_feats_path, metric, device)


if __name__ == "__main__":
    parser = arg_parser()
    parser.add_argument('--result_path', help='id features path')
    parser.add_argument('--train_feats_path', help='train features path')
    parser.add_argument('--ood_feats_path', help='ood features path')
    parser.add_argument('--id_feats_path', help='id features path')
    parser.add_argument("--num_of_clusters", required=True)
    parser.add_argument('--seed', default=0, type=int, help='set seed')
    parser.add_argument('--gmm_random_state', default=0, type=int, help='gmm random state')
    parser.add_argument("--clustering_model_path", required=True)
    parser.add_argument("--adjust_type", required=True)
    parser.add_argument("--adjust_scale", required=True)
    
    main(parser.parse_args())
