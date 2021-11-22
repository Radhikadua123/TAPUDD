from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

import pickle

class PlaneDataset(Dataset):
    def __init__(self):
        data_dir='./data/plane.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([0])
        y = torch.squeeze(y)
        return x, y

class PlaneDataset10Dim(Dataset):
    def __init__(self):
        data_dir='./data/plane_10dim.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([0])
        y = torch.squeeze(y)
        return x, y
    
class OvalDataset(Dataset):
    def __init__(self, split = 'train'):
        data_dir='./data/data_oval.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
            
        self.X =  self.data['X'][split]
        self.y =  self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y


class MoonDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/data_moons.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y

class CircleDataset(Dataset):
    def __init__(self, split = 'train'):
        data_dir='./data/data_circles.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
            
        self.X =  self.data['X'][split]
        self.y =  self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y


class FlowerDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/data_flowers.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y
    
class MultiOvalDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/iddata_2d.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y
    
class MultiOvalDataset10Dim(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/iddata_2d_dim10.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y
    
    
    
class MultiCircleDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/iddata_2d_cov01.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y
    
class MultiCircleDataset10Dim(Dataset):
    def __init__(self, split='train'):
        data_dir = './data/iddata_2d_cov01_dim10.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X'][split]
        self.y = self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y
    
class EllipseBinaryDataset(Dataset):
    def __init__(self, split = 'train'):
        data_dir='./data/iddata_2d_ellipse_binary.pkl'
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
            
        self.X = self.data['X'][split]
        self.y =  self.data['y'][split]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])
        y = torch.squeeze(y)
        return x, y 
    
def get_planeloader(input_dim):
    if input_dim == 10:
        data = PlaneDataset10Dim()
    elif input_dim == 2:
        data = PlaneDataset()
    else:
        raise NotImplementedError
    
    loader = DataLoader(data, batch_size=128, shuffle=False)
    return loader


def get_dataloaders(data_type):
    if data_type == 'oval':
        data_train = OvalDataset(split='train')
        data_valid = OvalDataset(split='val')
        data_test = OvalDataset(split='test')
    elif data_type == 'moons':
        data_train = MoonDataset(split='train')
        data_valid = MoonDataset(split='val')
        data_test = MoonDataset(split='test')
    elif data_type == 'circles':
        data_train = CircleDataset(split='train')
        data_valid = CircleDataset(split='val')
        data_test = CircleDataset(split='test')
    elif data_type =='flowers':
        data_train = FlowerDataset(split='train')
        data_valid = FlowerDataset(split='val')
        data_test = FlowerDataset(split='test')
    elif data_type == 'multioval':
        data_train = MultiOvalDataset(split='train')
        data_valid = MultiOvalDataset(split='val')
        data_test = MultiOvalDataset(split='test')
    elif data_type == 'multioval_10dim':
        data_train = MultiOvalDataset10Dim(split='train')
        data_valid = MultiOvalDataset10Dim(split='val')
        data_test = MultiOvalDataset10Dim(split='test')
    elif data_type == 'multicircle':
        data_train = MultiCircleDataset(split='train')
        data_valid = MultiCircleDataset(split='val')
        data_test = MultiCircleDataset(split='test')
    elif data_type == 'multicircle_10dim':
        data_train = MultiCircleDataset10Dim(split='train')
        data_valid = MultiCircleDataset10Dim(split='val')
        data_test = MultiCircleDataset10Dim(split='test')
    elif data_type == 'ellipse_binary':
        data_train = EllipseBinaryDataset(split='train')
        data_valid = EllipseBinaryDataset(split='val')
        data_test = EllipseBinaryDataset(split='test')
    else:
        raise NotImplementedError

    train_loader = DataLoader(data_train, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(data_valid, batch_size=128, shuffle=False, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=128, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": valid_loader,
        "test": test_loader,
    }
    return dataloaders












