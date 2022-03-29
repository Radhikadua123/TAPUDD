import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import torchvision.transforms as transforms
import torch
import os
import cv2
import skimage as sk
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def gaussian_noise(x, severity=1):
    """Function to add gaussian noise in images with different level of severity"""
    c = [0, .01, .02, .03, 0.04, .05, 0.06, .07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    """Function to add shot noise in images with different level of severity"""
    c = [0, .01, .02, .03, 0.04, .05, 0.06, .07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1][severity - 1]
    c = np.array(c)
    c = c * 2

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    """Function to add impulse noise in images with different level of severity"""
    c = [0, .01, .02, .03, 0.04, .05, 0.06, .07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


class ToTensor(object):
    
    def __call__(self, sample, size = 500):
        image, gender, label = sample['image'], sample['gender'], sample['label']
        image = cv2.resize(image,(size,size))
        image = np.expand_dims(image,axis = 0)
        image = torch.from_numpy(image)
        image = image.repeat(3, 1, 1)
        """ we need to convert  cuda.longtensors to cuda.floatTensor data type"""
        return {'image': image.float(),
                'gender': torch.from_numpy(gender).long(),
                'label':torch.from_numpy(label).float()}        


class Normalize(object):
    """Normalize images and bone age"""

    def __init__(self,img_mean,img_std,age_min,age_max):
        self.mean = img_mean
        self.std = img_std
        self.age_min = age_min
        self.age_max = age_max
    
    def __call__(self,sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['label']
        image -= self.mean
        image /= self.std
        bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)
        return {'image': image,
                'gender': gender,
                'label':bone_age} 


class BoneDataset(Dataset):
    """Custom Dataset for loading RSNA Boneage dataset."""

    def __init__(self, dataframe, img_dir, mode ='train', transform=None, age = [10,11,12]):
    
        df = dataframe
        df['path'] = df['id'].map(lambda x: os.path.join(img_dir,
                                                        '{}.png'.format(x)))
        df['gender'] = df['male'].map(lambda x: 1 if x else 0)
        self.img_dir = img_dir
        if(mode == 'train'):
            """Using 1500 images males and females during training. Total training images = 1500 Males + 1500 Females."""
            l = 1500
        else:
            """Using 200 images males and females during validation and testing. Total validation/test images = 200 Males + 200 Females."""
            l = 200

        # Use this to train model. Removing skewing of dataset by considering equal no of males and females.
        df1 = df.groupby(['gender']).get_group(0)
        df2 = df.groupby(['gender']).get_group(1)
        print("males and females", len(df1), len(df2))
        minimum_samples =  min(len(df1), len(df2)) #### number of samples from each group to remove skewedness

        df1 = df.groupby(['gender']).get_group(0)[:minimum_samples]
        df2 = df.groupby(['gender']).get_group(1)[:minimum_samples]
        print("males and females", len(df1), len(df2))

        # Combine images from positive and negative class(males and females) and shuffle the data.
        frames = [df1, df2]
        df3 = pd.concat(frames)
        df3 = df3.sample(frac = 1)

        self.img_names = df3['id'].values
        self.y = df3['boneage']
        self.gender = df3['gender']
        self.transform = transform

    def __getitem__(self, index):
        img = self.img_dir + str(self.img_names[index]) + '.png'
        img = cv2.imread(img,0)
        img = img.astype(np.float64)
        label = np.atleast_1d(self.y.values[index].astype('float'))
        gender = np.atleast_1d(self.gender.values[index].astype('float'))
        sample = {'image': img, 'gender':gender, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            image, gender, age = sample['image'], sample['gender'], sample['label']
        return image, gender

    def __len__(self):
        return self.y.shape[0]
    
    def print_bok(self):
        print(self.y)
    

class BoneDataset_adjust(Dataset):
    """Custom Dataset for loading RSNA Boneage dataset with distibution shift caused by the variation of brightness, contrast, shot noise,
       gaussian noise or impulse noise. Parameter "adjust_type" specifies the variation factor and parameter "adjust_scale" specifies the 
       severity with which the image should be shifted. """
    def __init__(self, dataframe, img_dir, transform=None, age = [10,11,12], adjust_type = 'bright', adjust_scale = 1):
    
        df = dataframe
        df['path'] = df['id'].map(lambda x: os.path.join(img_dir,
                                                        '{}.png'.format(x))) 
        df['gender'] = df['male'].map(lambda x: 1 if x else 0)
        self.img_dir = img_dir
        """Using 200 images males and females in OOD data. Total images = 200 Males + 200 Females."""
        df1 = df.groupby(['gender']).get_group(0)
        df2 = df.groupby(['gender']).get_group(1)
        print("males and females", len(df1), len(df2))
        minimum_samples =  min(len(df1), len(df2)) #### number of samples from each group to remove skewedness

        df1 = df.groupby(['gender']).get_group(0)[:minimum_samples]
        df2 = df.groupby(['gender']).get_group(1)[:minimum_samples]
        print("males and females", len(df1), len(df2))
        # Combine images from positive and negative class(males and females) and shuffle the data.
        frames = [df1, df2]
        df3 = pd.concat(frames)
        df3 = df3.sample(frac = 1)

        self.img_names = df3['id'].values
        self.y = df3['boneage']
        self.gender = df3['gender']
        self.transform = transform
        self.adjust_type = adjust_type
        self.adjust_scale = adjust_scale

    def __getitem__(self, index):
        img = self.img_dir + str(self.img_names[index]) + '.png'
        img = cv2.imread(img,0)
        # img = img.astype(np.float64)
        label = np.atleast_1d(self.y.values[index].astype('float'))
        gender = np.atleast_1d(self.gender.values[index].astype('float'))
        img_pil = Image.fromarray(img)

        """Generating distributionallly shifted data based on the variation factor."""
        if self.adjust_type=='bright':
            img = transforms.functional.adjust_brightness(img_pil, brightness_factor = self.adjust_scale)
        elif self.adjust_type == 'contrast':
            img = transforms.functional.adjust_contrast(img_pil, contrast_factor = self.adjust_scale)
        elif self.adjust_type == 'impulse_noise':
            img = impulse_noise(img_pil, self.adjust_scale)
        elif self.adjust_type == 'gaussian_noise':
            img = gaussian_noise(img_pil, self.adjust_scale)
        elif self.adjust_type == 'shot_noise':
            img = shot_noise(img_pil, self.adjust_scale)
        else:
            print("unavailable adjust_type")

        img = np.asarray(img)
        img = img.astype(np.float64)
        sample = {'image': img, 'gender':gender, 'label': label}

        if self.transform:
            sample = self.transform(sample)
            image, gender, age = sample['image'], sample['gender'], sample['label']
        return image, gender

    def __len__(self):
        return self.y.shape[0]
    
    def print_bok(self):
        print(self.y)

    
def get_adjust_dataloaders(bones_df, train_df, val_df, test_df, img_dir, data_transform, adjust = 'bright'):
    """ Dataloader function to obtain datasets (distibutionally shifted by variation given in the parameter "adjust". """

    """ adjust_scale stores the severity level to shift the dataset. """
    if adjust == 'bright':
        adjust_scale = [1]
        adjust_scale += list(np.arange(0,20,1)/10)
        adjust_scale += list(np.arange(20,80,5)/10)
        
    elif adjust == 'contrast':
        adjust_scale = [1]
        adjust_scale += list(np.arange(0,20,1)/10)
        adjust_scale += list(np.arange(20,80,5)/10)

    elif adjust == 'impulse_noise' or adjust == 'gaussian_noise' or adjust == 'shot_noise':
        adjust_scale = [1]
        adjust_scale += list(range(1,21))
    

    loaders = []
    data_len = []
    ind = BoneDataset_adjust(dataframe = test_df, img_dir = img_dir, transform = data_transform, age = [10,11,12], adjust_type = adjust, adjust_scale = adjust_scale[0])
    ind_len = len(ind)
    data_len.append(ind_len)
#     ind_loader = DataLoader(ind, batch_size=ind_len, shuffle=True)
    # ind_loader = DataLoader(ind, batch_size=256, shuffle=False, pin_memory=True, drop_last=False)
    ##use this for odin and godin eval
    ind_loader = DataLoader(ind, batch_size=16, shuffle=False, pin_memory=True, drop_last=False)
    
    loaders.append(ind_loader)
    
    for adjustness in adjust_scale[1:]:
        if(adjustness == 1.0):
            data_len.append(ind_len)
            loaders.append(ind_loader)
        else:
            ood = BoneDataset_adjust(dataframe = test_df, img_dir = img_dir, transform = data_transform, age = [10,11,12], adjust_type = adjust, adjust_scale = adjustness)
            ood_len = len(ood)
            data_len.append(ood_len)
    #         ood_loader = DataLoader(ood, batch_size=ood_len, shuffle=True)
            # ood_loader = DataLoader(ood, batch_size=256, shuffle=False, pin_memory=True, drop_last=False)
            ##use this for odin and godin eval
            ood_loader = DataLoader(ood, batch_size=16, shuffle=False, pin_memory=True, drop_last=False)
            loaders.append(ood_loader)
    return loaders, data_len, adjust_scale
    

def get_eval_dataloaders(bones_df, train_df, val_df, test_df, img_dir, data_transform, age_groups):
    """Dataloader function to obtain datasets (distributionally shifted by age). Parameter age_groups specifies the age bins for distributionally shifted datasets."""
    loaders = []
    data_len = []
    ind = BoneDataset(dataframe = test_df, img_dir = img_dir, mode = 'test', transform = data_transform)
    ind_len = len(ind)
#     ind_loader = DataLoader(ind, batch_size=ind_len, shuffle=False)

    ind_loader = DataLoader(ind, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
    age_grps = age_groups
    n = len(age_grps)
    for i in range(n):
        if(age_grps[i] == [10,11,12]):
            data_len.append(ind_len)
            loaders.append(ind_loader)
        else:
            ood = BoneDataset(dataframe = bones_df, img_dir = img_dir, mode = 'test', transform = data_transform, age = age_grps[i])
            ood_len = len(ood)
            data_len.append(ood_len)
#             ood_loader = DataLoader(ood, batch_size=ood_len, shuffle=False)
            ood_loader = DataLoader(ood, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
            loaders.append(ood_loader)
    return loaders, data_len
