import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import log
from utils.test_utils import arg_parser
from utils.test_utils import stable_cumsum, fpr_and_fdr_at_recall, get_measures, plot_aupr_auroc

def MahalaEnsemble(args, logger, clusters, logdir, reverse, best):
    
    for cluster in clusters:
        file_path_ood_scores = os.path.join(args.ood_scores_path, f"cluster{cluster}", "ood_scores.npy")
        file_path_id_scores = os.path.join(args.id_scores_path, f"cluster{cluster}", "ood_scores.npy")
        in_examples = np.load(file_path_id_scores)
        out_examples = np.load(file_path_ood_scores)
        examples = np.squeeze(np.vstack((in_examples, out_examples)))
        if(cluster == 1):
            gmm_Df = pd.DataFrame(zip(examples), columns=["cluster_1"])
        else:
            name1 = f"cluster{cluster}"
            gmm_Df[name1] = examples

    labels = np.zeros(len(in_examples) + len(out_examples), dtype=np.int32)
    labels[:len(in_examples)] += 1
    length = len(in_examples)
    EnsembleTop8(logger, args, labels, gmm_Df, length, reverse = reverse, best = best)
                


def EnsembleTop8(logger, args, labels, gmm_Df, length_in_egs, reverse=False, best=False):
    
    logger.flush()
    df_top_8_clus = gmm_Df
    cols = list(gmm_Df.columns)
    std = gmm_Df.std(axis=1)
    mean = gmm_Df.mean(axis=1)
    min1 = gmm_Df.min(axis=1)
    max1 = gmm_Df.max(axis=1)

    if(args.average == "True"):
        ##### Ensemble using mean of all models ######
        print("AVERAGE")
        df_final =  df_top_8_clus

    elif(args.extremes == "False"):
        ##### Dropping two models with least and maximum score and ensemble using mean of all models ######
        print("TRIM AVERAGE")
        df_final = df_top_8_clus.apply(np.sort, axis=1).apply(lambda x: x[2:10]).apply(pd.Series)
    
    elif(args.extremes == "True"):

        if(best == False):
            ##### Ensemble using mean of 8 models based on maximum or minimum MB distance ######
            if(reverse==True):
                print("TOP")
                ######## gmm_df_top 8_clusters based on highest MB distance #########
                df_top_8_clus = df_top_8_clus.apply(np.sort, axis=1).apply(lambda x: x[-8:]).apply(pd.Series)
            else:
                ######## gmm_df_top 8_clusters based on lowest MB distance #########
                print("BOTTOM")
                df_top_8_clus = df_top_8_clus.apply(np.sort, axis=1).apply(lambda x: x[:8]).apply(pd.Series)
            df_top_8_clus.columns = ['Top1', 'Top2', 'Top3','top4','top5','top6', 'top7','top8']

        else:
            ##### Ensemble using extreme models (similar to seesaw) ######
            print("SEESAW")
            gmm_Df["mean_MB"] = mean
            gmm_Df['more_than_mean'] = gmm_Df[cols].ge(gmm_Df["mean_MB"],axis=0).sum(axis=1)

            df_more_than_mean = gmm_Df.loc[gmm_Df['more_than_mean'] >= 6][cols]
            df_less_than_mean = gmm_Df.loc[gmm_Df['more_than_mean'] < 6][cols]

            df_more_than_mean = df_more_than_mean.apply(np.sort, axis=1).apply(lambda x: x[-8:]).apply(pd.Series)
            df_more_than_mean.columns = ['Top1', 'Top2', 'Top3','Top4','Top5','Top6', 'Top7','Top8']
            df_less_than_mean = df_less_than_mean.apply(np.sort, axis=1).apply(lambda x: x[:8]).apply(pd.Series)
            df_less_than_mean.columns = ['Top1', 'Top2', 'Top3','Top4','Top5','Top6', 'Top7','Top8']

            df_top_8_clus = pd.concat([df_more_than_mean, df_less_than_mean], ignore_index=False).sort_index()

        df_final = df_top_8_clus

    std = df_final.std(axis=1)
    mean = df_final.mean(axis=1)
    min1 = df_final.min(axis=1)
    max1 = df_final.max(axis=1)         
    ensemble_MB = mean

    df_final["std_MB"] = std
    df_final["mean_MB"] = mean
    df_final["min_MB"] = min1
    df_final["max_MB"] = max1
    df_final["ID sample"] = labels

    print("Min and max std of MB across 8 cluster for ID samples:", min(df_final["std_MB"][:length_in_egs]), max(df_final["std_MB"][:length_in_egs]))
    print("Min and max std of MB across 8 cluster for OOD samples:", min(df_final["std_MB"][length_in_egs:]), max(df_final["std_MB"][length_in_egs:]))

    print("Min and max mean of MB across 8 cluster for ID samples:", min(df_final["mean_MB"][:length_in_egs]), max(df_final["mean_MB"][:length_in_egs]))
    print("Min and max mean of MB across 8 cluster for OOD samples:", min(df_final["mean_MB"][length_in_egs:]), max(df_final["mean_MB"][length_in_egs:]))


    if(args.scaling == "False"):
        in_examples = np.array(df_final["mean_MB"][:length_in_egs]).reshape((-1, 1))
        out_examples = np.array(df_final["mean_MB"][length_in_egs:]).reshape((-1, 1))
    else:
        print("Scaling")
        in_examples = ((np.array(df_final["mean_MB"][:length_in_egs])-1) * (np.array(df_final["std_MB"][:length_in_egs])+1)).reshape((-1, 1))
        out_examples = ((np.array(df_final["mean_MB"][length_in_egs:])-1) * (np.array(df_final["std_MB"][length_in_egs:])+1)).reshape((-1, 1))
    

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    path_auroc_aupr = os.path.join(args.logdir, args.name)
    auc_roc, auc_aupr_in, auc_aupr_out = plot_aupr_auroc(in_examples, out_examples, path_auroc_aupr)


    hist_mean_ood = -np.log(-np.array(mean[length_in_egs:]) + 2)
    hist_mean_id = -np.log(-np.array(mean[:length_in_egs]) + 2)

    hist_std_ood = np.log(np.array(std[length_in_egs:]) + 2)
    hist_std_id = np.log(np.array(std[:length_in_egs]) + 2)

    hist_scaled_mean_ood = -np.log(-np.array(mean[length_in_egs:]) + 2) * np.log(np.array(std[length_in_egs:]) + 2)
    hist_scaled_mean_id = -np.log(-np.array(mean[:length_in_egs]) + 2) * np.log(np.array(std[:length_in_egs]) + 2)

    print("before min and max id", min(hist_mean_id), max(hist_mean_id))
    print("before min and max ood", min(hist_mean_ood), max(hist_mean_ood))

    print("after min and max id", min(hist_scaled_mean_id), max(hist_scaled_mean_id))
    print("after min and max ood", min(hist_scaled_mean_ood), max(hist_scaled_mean_ood))

    plt.figure(figsize=(5, 6))
    plt.hist(hist_mean_id, bins='auto', label = "ID", alpha = 0.5)
    plt.hist(hist_mean_ood, bins='auto', label = "OOD", alpha = 0.5)
    plt.legend()
    plt.title("Histogram of MEAN_MB")
    plt.xlim([-20, -5])
    plt.ylim([0, 2000])
    path1 = os.path.join(path_auroc_aupr, 'hist_mean.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)

    plt.figure(figsize=(5, 6))
    plt.hist(hist_std_id, bins='auto', label = "ID", alpha = 0.5)
    plt.hist(hist_std_ood, bins='auto', label = "OOD", alpha = 0.5)
    plt.legend()
    plt.title("Histogram of STD_MB")
    plt.xlim([-1, 15])
    plt.ylim([0, 2000])
    path1 = os.path.join(path_auroc_aupr, 'hist_std.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)

    plt.figure(figsize=(5, 6))
    plt.hist(hist_scaled_mean_id, bins='auto', label = "ID", alpha = 0.5)
    plt.hist(hist_scaled_mean_ood, bins='auto', label = "OOD", alpha = 0.5)
    plt.ylim([0, 2000])
    plt.xlim([-80, -5])
    plt.legend()
    plt.title("Histogram of Scaled MEAN_MB")
    path1 = os.path.join(path_auroc_aupr, 'hist_scaled_mean.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)
    logger.info('=================== Results for GMM + Mahala =================')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))
    logger.info('Recalculating AUROC using sk.auc: {}'.format(auc_roc))
    logger.info('Recalculating AUPR (In) using sk.auc: {}'.format(auc_aupr_in))
    logger.info('Recalculating AUPR (Out) using sk.auc: {}'.format(auc_aupr_out))

    logger.flush()


def main(args):
    logger = log.setup_logger(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if(args.top == "True"):
        reverse = True
    else: 
        reverse = False
    
    if(args.best == "True"):
        best = True
    else: 
        best = False
    clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32]
    MahalaEnsemble(args, logger, clusters, args.logdir, reverse, best)

if __name__ == "__main__":
    parser = arg_parser()
    parser.add_argument("--top", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--extremes", required=True)
    parser.add_argument("--average", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument('--ood_scores_path', help='path to ood scores')
    parser.add_argument('--id_scores_path', help='path to id scores') 

    main(parser.parse_args())