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


import plotly
import plotly as py
import seaborn as sns
from torch.utils.data import Dataset
import argparse
from utils import log
from utils.test_utils import arg_parser
from matplotlib.patches import Ellipse
import plotly.graph_objs as go
from scipy.stats import gaussian_kde
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def visualization_density_hexbin(args, train_Df, oods, path1):
    total_df = []
    train_feats = train_Df["feats"].tolist()
    total_df.extend(train_feats)
    
    start = 1
    plt.figure()
    ax = plt.gca()
    
    pca_2d = PCA(n_components=2, random_state=0)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(total_df))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    colors = sns.color_palette("tab10")

    train_Df1 = pd.concat([train_Df,PCs_2d], axis=1, join='inner')
    train_Df1["dummy"] = 0

    if (path1 == f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        name1 =  "ID data"
        ax.hexbin(train_Df1["PC1_2d"], train_Df1["PC2_2d"], gridsize = 100)
    # fig = ax.scatter(x, y, c=z, s=0.03, marker = "x", label = name1)
    # plt.colorbar(fig, label='point density') 

    ood1 = oods["feats"].tolist()
    
    if (path1 != f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        PCs_2d = pd.DataFrame(pca_2d.transform(ood1))
        PCs_2d.columns = ["PC1_2d", "PC2_2d"]
        ood_df1 = pd.concat([oods, PCs_2d], axis=1, join='inner')
        ood_df1["dummy"] = 0
        
        name1 =  "OOD data"
        ax.hexbin(ood_df1["PC1_2d"], ood_df1["PC2_2d"], gridsize = 100)
        
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.xlim(-300, 800)
    plt.ylim(-300, 500)
    plt.grid()
    
    lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    # lgnd.legendHandles[0]._sizes = [30]
    # if (path1 != '/home/radhika/mos1/checkpoints/pca_plots_density_separate/imagenet.png'):
    #     lgnd.legendHandles[1]._sizes = [30]
    plt.savefig(path1, bbox_inches='tight', dpi=300)


def visualization_density_histogram(args, train_Df, oods, path1):
    total_df = []
    train_feats = train_Df["feats"].tolist()
    total_df.extend(train_feats)
    
    start = 1
    plt.figure()
    ax = plt.gca()
    
    pca_2d = PCA(n_components=2, random_state=0)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(total_df))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    colors = sns.color_palette("tab10")

    train_Df1 = pd.concat([train_Df,PCs_2d], axis=1, join='inner')
    train_Df1["dummy"] = 0

    if (path1 == f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        name1 =  "ID data"
        ax.hist2d(train_Df1["PC1_2d"], train_Df1["PC2_2d"], bins=100)
    # fig = ax.scatter(x, y, c=z, s=0.03, marker = "x", label = name1)
    # plt.colorbar(fig, label='point density') 

    ood1 = oods["feats"].tolist()
    
    if (path1 != f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        PCs_2d = pd.DataFrame(pca_2d.transform(ood1))
        PCs_2d.columns = ["PC1_2d", "PC2_2d"]
        ood_df1 = pd.concat([oods, PCs_2d], axis=1, join='inner')
        ood_df1["dummy"] = 0

        name1 =  "OOD data"
        ax.hist2d(ood_df1["PC1_2d"], ood_df1["PC2_2d"], bins=100)
        
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.xlim(-300, 800)
    plt.ylim(-300, 500)
    plt.grid()
    
    lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    # lgnd.legendHandles[0]._sizes = [30]
    # if (path1 != '/home/radhika/mos1/checkpoints/pca_plots_density_separate/imagenet.png'):
    #     lgnd.legendHandles[1]._sizes = [30]
    plt.savefig(path1, bbox_inches='tight', dpi=300)


def visualization_density_gaussian_kde(args, train_Df, oods, path1):
    total_df = []
    train_feats = train_Df["feats"].tolist()
    total_df.extend(train_feats)
    
    start = 1
    plt.figure()
    ax = plt.gca()
    
    pca_2d = PCA(n_components=2, random_state=0)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(total_df))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    colors = sns.color_palette("tab10")

    train_Df1 = pd.concat([train_Df,PCs_2d], axis=1, join='inner')
    train_Df1["dummy"] = 0
    # Calculate the point density
    xy = np.vstack([train_Df1["PC1_2d"], train_Df1["PC2_2d"]])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = train_Df1["PC1_2d"][idx], train_Df1["PC2_2d"][idx], z[idx]

    if (path1 == f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        name1 =  "ID data"
        # fig = ax.scatter(x, y, c=z, s=0.04, marker = "o", label = name1, cmap="Reds")
        fig = ax.scatter(x, y, c=z, s=0.03, marker = "o", cmap="Reds")
        plt.colorbar(fig, label='density') 

    ood1 = oods["feats"].tolist()
    
    if (path1 != f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
        
        if(args.separate == "False"):
            name1 =  "ID data"
            # ax.scatter(x, y, c=z, s=0.04, marker = "o", label = name1, cmap="Reds")
            ax.scatter(x, y, c=z, s=0.05, marker = "o", cmap="Reds")
            # plt.colorbar(fig, label='density') 
        PCs_2d = pd.DataFrame(pca_2d.transform(ood1))
        PCs_2d.columns = ["PC1_2d", "PC2_2d"]
        ood_df1 = pd.concat([oods, PCs_2d], axis=1, join='inner')
        ood_df1["dummy"] = 0
        xy = np.vstack([ood_df1["PC1_2d"], ood_df1["PC2_2d"]])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = ood_df1["PC1_2d"][idx], ood_df1["PC2_2d"][idx], z[idx]

        name1 =  "OOD data"
        # ax.scatter(x, y, c=z, s=0.04, marker = "o", label = name1, cmap="Blues")
        if(args.separate == "True"):
            fig = ax.scatter(x, y, c=z, s=0.07, marker = ".",cmap="Blues")
            plt.colorbar(fig, label='density') 
        else:
            ax.scatter(x, y, c=z, s=0.07, marker = ".",cmap="Blues")
            
        
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.xlim(-300, 600)
    plt.ylim(-300, 450)
    plt.grid()
    
    # lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    # if (args.separate == "False"):
    #     lgnd.legendHandles[0]._sizes = [30]
    #     lgnd.legendHandles[0].set_color("red")
    #     lgnd.legendHandles[1]._sizes = [30]
    #     lgnd.legendHandles[1].set_color("blue")
    # else:
    #     if (path1 == f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate/imagenet.png'):
    #         lgnd.legendHandles[0]._sizes = [30]
    #         lgnd.legendHandles[0].set_color("red")
    #     else:
    #         lgnd.legendHandles[0]._sizes = [30]
    #         lgnd.legendHandles[0].set_color("blue")
    plt.savefig(path1, bbox_inches='tight', dpi=300)


#Instructions for building the 2-D plot
def visualization(train_Df, oods, path1):
    total_df = []
    train_feats = train_Df["feats"].tolist()
    total_df.extend(train_feats)
    
    start = 1
    plt.figure()
    ax = plt.gca()
    
    pca_2d = PCA(n_components=2, random_state=0)
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(total_df))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    colors = sns.color_palette("tab10")

    train_Df1 = pd.concat([train_Df,PCs_2d], axis=1, join='inner')
    train_Df1["dummy"] = 0

    name1 =  "ID data"
    ax.scatter(train_Df1["PC1_2d"], train_Df1["PC2_2d"], c = "blue", s = 0.03, label = name1, marker = "o", alpha=0.4)

    ood1 = oods["feats"].tolist()
    
    if (path1 != '/home/radhika/mos1/checkpoints/pca_plots/imagenet.png'):
        PCs_2d = pd.DataFrame(pca_2d.transform(ood1))
        PCs_2d.columns = ["PC1_2d", "PC2_2d"]
        ood_df1 = pd.concat([oods, PCs_2d], axis=1, join='inner')
        ood_df1["dummy"] = 0
        name1 =  "OOD"
        ax.scatter(ood_df1["PC1_2d"], ood_df1["PC2_2d"], c = "magenta", s = 0.03, label = name1, marker = "o", alpha=0.4)
        
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.xlim(-300, 800)
    plt.ylim(-300, 500)
    plt.grid()
    lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    lgnd.legendHandles[0]._sizes = [30]
    if (path1 != '/home/radhika/mos1/checkpoints/pca_plots/imagenet.png'):
        lgnd.legendHandles[1]._sizes = [30]
    plt.savefig(path1, bbox_inches='tight', dpi=300)
    

def main(args):
    # logger = log.setup_logger(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ood_feats = []
    in_feats = []

    in_feats.extend(np.load("features/id_data/feats.npy"))
    ood_feats.extend(np.load(args.ood_feats_path))

    in_Df = pd.DataFrame(list(zip(in_feats)), columns=["feats"])
    ood_Df = pd.DataFrame(list(zip(ood_feats)), columns=["feats"])

    if(args.density == "True"):
        if(args.separate == "True"):
            pca_plots_path = f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}_separate'
        else:
            pca_plots_path = f'/home/radhika/mos1/checkpoints/pca_plots_density_{args.density_type}'
    else:
        pca_plots_path = f'/home/radhika/mos1/checkpoints/pca_plots'
    os.makedirs(pca_plots_path, exist_ok=True)

    if(args.separate == "True"):
        pca_plots_path_file = os.path.join(pca_plots_path, 'imagenet.png')
        if(args.density == "True"):
            if(args.density_type == "gaussian_kde"):
                visualization_density_gaussian_kde(args, in_Df, ood_Df, pca_plots_path_file)
            if(args.density_type == "hexbin"):
                visualization_density_hexbin(args, in_Df, ood_Df, pca_plots_path_file)
            if(args.density_type == "histogram"):
                visualization_density_histogram(args, in_Df, ood_Df, pca_plots_path_file)
        else:
            visualization(in_Df, ood_Df, pca_plots_path_file)

    pca_plots_path_file = os.path.join(pca_plots_path, f'{args.dataset_name}.png')
    print(args.separate, args.density, pca_plots_path_file)
    if(args.density == "True"):
        if(args.density_type == "gaussian_kde"):
            visualization_density_gaussian_kde(args, in_Df, ood_Df, pca_plots_path_file)
        if(args.density_type == "hexbin"):
            visualization_density_hexbin(args, in_Df, ood_Df, pca_plots_path_file)
        if(args.density_type == "histogram"):
            visualization_density_histogram(args, in_Df, ood_Df, pca_plots_path_file)
    else:
        visualization(in_Df, ood_Df, pca_plots_path_file)


if __name__ == "__main__":
    parser = arg_parser()
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument('--ood_feats_path', help='path to tuned mahalanobis parameters')
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--density_type", required=True)
    parser.add_argument("--separate", required=True)
    main(parser.parse_args())