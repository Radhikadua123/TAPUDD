import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import cv2

from utils import *
from dataset import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

features = None

def get_features_hook(self, input, output):
    global features
    features = [output]

def get_features(model, data, num_classes, device=None):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    handle = model.fc[1].register_forward_hook(get_features_hook)
    model(data)
    handle.remove()
    global features
    out_features = features[0]
    
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2) #N, 128

    return out_features


def get_trainftrs(model, args, train_loader = None, device = None):
    """ Generates penultimate layer features, target and predicted label. """
    model = model.to(args.device)
    model.eval()

    total = 0
    correct = 0

    value_inds = []
    value_oods = []
    names_oods = []
    
    features = []
    targets = []
    preds = []
    i = 0 
    
    for idx_ins, data in tqdm(enumerate(train_loader)):
        print(i)
        images, labels = data

        images = images.to(args.device)
        labels = labels.to(args.device)
        print("args.device", args.device)

        with torch.no_grad():
            #forward
            feature_small = get_features(model, images, 2) #N, 128
            outputs = model(images) 
            predicted_value, predicted = torch.max(outputs.data, 1)
            trgs_small = labels
            preds_small = predicted

        features.append(feature_small)
        targets.append(trgs_small)
        preds.append(preds_small)
        i += 1

    features_all = torch.cat(features)
    trgs_all = torch.cat(targets)    
    preds_all = torch.cat(preds)  
    dir_path = os.path.join(args.result_path, "seed_" + str(args.seed), "penultimate_ftrs")
    os.makedirs(dir_path, exist_ok=True)

    if args.flag_adjust:
        file_path = os.path.join(dir_path, "ftrs_{}_{}.npy".format(args.variation, 'train'))
        trg_pth = os.path.join(dir_path, "trgs_{}_{}.npy".format(args.variation, 'train'))
        preds_pth = os.path.join(dir_path, "preds_{}_{}.npy".format(args.variation, 'train'))
    else:
        file_path = os.path.join(dir_path, "ftrs_age_{}.npy".format('train'))
        trg_pth = os.path.join(dir_path, "trgs_age_{}.npy".format('train'))
        preds_pth = os.path.join(dir_path, "preds_age_{}.npy".format('train'))
    
    np.save(file_path, features_all.detach().cpu().numpy())
    np.save(trg_pth, trgs_all.detach().cpu().numpy())
    np.save(preds_pth, preds_all.detach().cpu().numpy())

            
def test(model, args, train_loader = None, loaders = None, device = None,train_loader_mu=None):
    """ Generates penultimate layer features, target and predicted label for distributionally shifted datasets. """
    model = model.to(args.device)
    model.eval()

    total = 0
    correct = 0
    dict_results = dict()
    dict_results['preds'] = []
    dict_results['trues'] = []
    dict_results['correct'] = []
    dict_results['dataset_idx'] = []
    dict_results['org_labels'] = []
    dict_results['pred_labels'] = []
    value_inds = []
    value_oods = []
    names_oods = []
    
    test_loaders = loaders
    print(len(test_loaders))
    
    for idx, loader in enumerate(test_loaders):
        features = []
        targets = []
        preds = []
        for idx_ins, data in tqdm(enumerate(loader)):
            images, labels = data
            
            images = images.to(args.device)
            labels = labels.to(args.device)
            #forward
            with torch.no_grad():
                feature_small = get_features(model, images, 2) #N, 128   
                outputs = model(images) 
                predicted_value, predicted = torch.max(outputs.data, 1)
                preds_small = predicted

            trgs_small = labels
            features.append(feature_small)
            targets.append(trgs_small)
            preds.append(preds_small)

        features_all = torch.cat(features)
        trgs_all = torch.cat(targets)    
        preds_all = torch.cat(preds)        

        dir_path = os.path.join(args.result_path, "seed_" + str(args.seed), "penultimate_ftrs")
        os.makedirs(dir_path, exist_ok=True)
        if args.flag_adjust:
            file_path = os.path.join(dir_path, "ftrs_{}_{}.npy".format(args.variation, idx))
            trg_pth = os.path.join(dir_path, "trgs_{}_{}.npy".format(args.variation, idx))
            preds_pth = os.path.join(dir_path, "preds_{}_{}.npy".format(args.variation, idx))
        else:
            file_path = os.path.join(dir_path, "ftrs_age_{}.npy".format(idx))
            trg_pth = os.path.join(dir_path, "trgs_age_{}.npy".format(idx))
            preds_pth = os.path.join(dir_path, "preds_age_{}.npy".format(idx))

        np.save(file_path, features_all.detach().cpu().numpy())
        np.save(trg_pth, trgs_all.detach().cpu().numpy())
        np.save(preds_pth, preds_all.detach().cpu().numpy())

    vector_pth = os.path.join(dir_path, "class_vectors.npy".format(idx))
    np.save(vector_pth, model.fc[2].weight.detach().cpu().numpy())

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag_adjust', action='store_true', help='adjust test or not')
    parser.add_argument('--variation', type=str, help='bright or contrast')
    parser.add_argument('--num_classes', default = 2, type=int, help='path of the model')
    parser.add_argument('--result_path', default="./results", type=str, help='train or test')
    parser.add_argument('--seed', default = 0, type=int, help='path of the model')
    parser.add_argument('--data_path', default="./data", type=str, help='path of the dataset')
    args = parser.parse_args()

    set_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##############################
    # Obtain OOD datasets
    ##############################
    bones_df, train_df, val_df, test_df, data_transform = Data_Transform(args.data_path)
    images_dir = os.path.join(args.data_path, 'boneage-training-dataset/boneage-training-dataset/')
    age_groups = [[1,2,3,4,5],[6],[7],[8],[9],[10,11,12],[13],[14],[15,16,17,18,19]]
    if args.flag_adjust:
        """ Obtain loaders of OOD datasets shifted by variation of gaussian, impulse, shot noise or brightness, contrast. """
        loaders, data_len, adjust_scale = get_adjust_dataloaders(bones_df, train_df, val_df, test_df, images_dir, data_transform, args.variation)
    
    else:
        """ Obtain loaders of OOD datasets shifted by the varaition of age. """
        loaders, data_len = get_eval_dataloaders(bones_df, train_df, val_df, test_df, images_dir, data_transform, age_groups)

    
    train_dataset = BoneDataset(dataframe = train_df, img_dir = images_dir, mode = 'train', transform = data_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_loader_mu = DataLoader(train_dataset, batch_size=64, shuffle=False)

    ##############################
    # Define model and load weights
    ##############################
    model = define_model(device)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    model.eval()
    
    ###############################
    # Obtain penultimate layer features for ood and train datasets.
    ###############################
    test(model, args, train_loader = train_loader, loaders = loaders, device = device, train_loader_mu = train_loader_mu)

    dir_path = os.path.join(args.result_path, "seed_" + str(args.seed), "penultimate_ftrs")
    os.makedirs(dir_path, exist_ok=True)

    if args.flag_adjust:
        file_path = os.path.join(dir_path, "ftrs_{}_{}.npy".format(args.variation, 'train'))
    else:
        file_path = os.path.join(dir_path, "ftrs_age_{}.npy".format('train'))
    
    if not os.path.exists(file_path):
        get_trainftrs(model, args, train_loader = train_loader, device = device)
            
if __name__ == '__main__':
    main()
