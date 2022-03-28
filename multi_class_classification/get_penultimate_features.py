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
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator, get_Mahalanobis_score
import torch.nn as nn
import pickle5 as pickle
from sklearn.linear_model import LogisticRegressionCV
from utils.test_utils import arg_parser, mk_id_ood

torch.cuda.empty_cache()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


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


def extract_features(model, loader, save_path):
    features = []

    for num_batch, (data, target) in enumerate(loader):
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
    np.save(os.path.join(save_path, 'feats.npy'), features_all.detach().cpu().numpy())
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
    dir_path = os.path.join("features", "imagenet", "train")
    os.makedirs(dir_path, exist_ok=True)
    extract_features(model, train_loader, dir_path)

    in_set, out_set, in_loader, out_loader = mk_id_ood(args, logger)

    dir_path = os.path.join("features", "id_data")
    os.makedirs(dir_path, exist_ok=True)
    extract_features(model, in_loader, dir_path)
    print("id data done ###########")

    dir_path = os.path.join("features", "ood_data", "textures")
    os.makedirs(dir_path, exist_ok=True)
    extract_features(model, out_loader, dir_path)


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

    main(parser.parse_args())
