import cv2
import glob
import torch
import random
import numpy as np 
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from dataset import *
from skimage import io, transform
from torchvision import datasets, models, transforms

def define_model(device = 'cuda'):
    """function to define the model"""
    model = models.resnet18(pretrained = True)
    model.fc = nn.Sequential(
              nn.Linear(in_features = 512, out_features = 128, bias = True),
              nn.ReLU(),
              nn.Linear(in_features = 128, out_features = 2, bias = True),
            )
    model = model.to(device)
    return model

def get_outputs(model, images, args):
    """ """
    outputs = model(images)
    logit = outputs
    outputs = nn.Softmax()(outputs)
    return logit, outputs


def set_seed(random_seed):
    """ function to set seed """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Data_Transform(path):
    """ function to return the test, val and train dataframes and the transform applied on it."""
    k = 100
    size = 500
    train_dataset_path =  os.path.join(path, 'boneage-training-dataset/boneage-training-dataset/')
    csv_path =  os.path.join(path, 'boneage-training-dataset.csv')
    ############## Find mean and std of dataset###################
    image_filenames = glob.glob(train_dataset_path+'*.png')
    random_images = random.sample(population = image_filenames, k = len(image_filenames))
    
#     means = []
#     stds = []

#     for filename in random_images:
#         image = cv2.imread(filename,0)
#         image = cv2.resize(image,(size,size))
#         mean,std = cv2.meanStdDev(image)
#     #    mean /= 255
#     #    std /= 255
        
#         means.append(mean[0][0])
#         stds.append(std[0][0])

#     avg_mean = np.mean(means) 
#     avg_std = np.mean(stds)

#     print('Approx. Mean of Images in Dataset: ',avg_mean)
#     print('Approx. Standard Deviation of Images in Dataset: ',avg_std)

    ############### After calculating mean and std of entire dataset from above, we get the below values ############
    avg_mean = 46.49
    avg_std = 42.56
    
    dataset_size = len(image_filenames)-2800
    val_size = dataset_size + 1400

    bones_df = pd.read_csv(csv_path)
    bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)
    bones_df['AgeM']=bones_df['boneage'].apply(lambda x: round(x/12.)).astype(int) #converting age from months to years

    train_df = bones_df.iloc[:dataset_size,:]
    val_df = bones_df.iloc[dataset_size:val_size,:]
    test_df = bones_df.iloc[val_size:,:]

    age_max = np.max(bones_df['boneage'])
    age_min = np.min(bones_df['boneage'])
    print("avg_mean, avg_std, age_min, age_max", avg_mean, avg_std, age_min, age_max)
    data_transform = transforms.Compose([
    Normalize(avg_mean, avg_std, age_min, age_max),
    ToTensor()   
    ]) 
    return bones_df, train_df, val_df, test_df, data_transform