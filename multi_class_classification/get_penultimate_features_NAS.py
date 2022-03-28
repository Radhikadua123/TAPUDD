"""Loads the images from validation data of imagenet (i.e, the ID data) and adjusts brightness/contrast of the images.  Then, the features of each of the NAS datasets are extracted"""

from utils import log
import resnetv2
import torch
import time
import torchvision as tv
import numpy as np
import argparse
from dataset import DatasetWithMeta
from utils.test_utils import get_measures
import os
import glob
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator, get_Mahalanobis_score
import torch.nn as nn
import pickle5 as pickle
from sklearn.linear_model import LogisticRegressionCV
from utils.test_utils import arg_parser
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import cv2

torch.cuda.empty_cache()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def gaussian_noise(x, severity=1):
    """Function to add gaussian noise in images with different level of severity"""
    c = [0, .01, .02, .03, 0.04, .05, 0.06, .07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


class Imagenet_adjust_NAS(Dataset):
    """Custom Dataset for loading RSNA Boneage dataset with distibution shift caused by the variation of brightness, contrast, shot noise,
       gaussian noise or impulse noise. Parameter "adjust_type" specifies the variation factor and parameter "adjust_scale" specifies the 
       severity with which the image should be shifted. """
    def __init__(self, img_dir, transform=None, adjust_type = 'bright', adjust_scale = 1):

        
        self.img_dir1 = os.path.join(img_dir, "**/*.JPEG")
        self.img_names = glob.glob(self.img_dir1, recursive=True)
        self.transform = transform
        self.adjust_type = adjust_type
        self.adjust_scale = adjust_scale

    def __getitem__(self, index):
        img = cv2.imread(self.img_names[index])
        img_pil = Image.fromarray(img)
        """Generating distributionallly shifted data based on the variation factor."""
        if self.adjust_type=='bright':
            img = tv.transforms.functional.adjust_brightness(img_pil, brightness_factor = self.adjust_scale)
        elif self.adjust_type == 'contrast':
            img = tv.transforms.functional.adjust_contrast(img_pil, contrast_factor = self.adjust_scale)
        elif self.adjust_type == 'gaussian_noise':
            img = gaussian_noise(img_pil, self.adjust_scale)
            img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            print("unavailable adjust_type")

        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_names)
    
    def print_bok(self):
        print(self.img_names)


def mk_id_ood(args, logger, adjust_type, adjust_scale):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    nas_set = Imagenet_adjust_NAS(args.in_datadir, transform = val_tx, adjust_type = adjust_type, adjust_scale = adjust_scale)
    print(f"Using an in-distribution set with {len(nas_set)} images.")

    nas_loader = torch.utils.data.DataLoader(
        nas_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return nas_set, nas_loader


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = DatasetWithMeta(args.datadir, args.train_list, val_tx)
    valid_set = DatasetWithMeta(args.datadir, args.val_list, val_tx)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_set, valid_set, train_loader, valid_loader

def extract_features(model, loader, save_path, adjust_scale):
    features = []

    for num_batch, (data) in enumerate(loader):
        if num_batch % 100 == 0:
            print('{} batches processed...'.format(num_batch))
    
        with torch.no_grad():
            data = Variable(data)
            data = data.cuda()
            out_features = [model(x=data, layer_index=4)]

        num_output = len(out_features)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).data.cpu()
        features.append(out_features[0])
    
    features_all = torch.cat(features)
    np.save(os.path.join(save_path, f'feats_adjust_scale_{adjust_scale}.npy'), features_all.detach().cpu().numpy())
    return

def main(args):
    logger = log.setup_logger(args)
    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    train_set, val_set, train_loader, val_loader = mktrainval(args, logger)

    logger.info(f"Loading model from {args.model_path}")
    model = resnetv2.KNOWN_MODELS[args.model](head_size=len(train_set.classes))
    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    adjust_type = args.adjust_type

    if adjust_type == 'bright':
        adjust_scale = [1.0]
        adjust_scale += list(np.arange(0,20,1)/10)
        adjust_scale += list(np.arange(20,80,5)/10)
        
    elif adjust_type == 'contrast':
        adjust_scale = [1]
        adjust_scale += list(np.arange(0,20,1)/10)
        adjust_scale += list(np.arange(20,80,5)/10)

    elif adjust_type == "gaussian_noise":
        adjust_scale = [1]
        adjust_scale += list(range(1,21))

    for adjust_scale_curr in adjust_scale:
        print("####BRIGHTNESS ######", adjust_scale_curr)
        nas_set, nas_loader = mk_id_ood(args, logger, adjust_type, adjust_scale_curr)
        dir_path = os.path.join("features", "nas_data_imagenet", adjust_type)
        os.makedirs(dir_path, exist_ok=True)
        extract_features(model, nas_loader, dir_path, adjust_scale_curr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", required=True)

    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--model", default="BiT-S-R101x1",
                        help="Which variant to use")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--logdir", required=True,
                        help="Where to log training info (small).")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")
    parser.add_argument("--adjust_type", required=True)

    main(parser.parse_args())