from utils import log
import resnetv2
import torch
import time
import glob 
import cv2
import os

from PIL import Image
import torchvision as tv
from dataset import DatasetWithMeta
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from utils.test_utils import arg_parser, mk_id_ood, get_measures, plot_aupr_auroc
from finetune import get_group_slices

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
            img = gaussian_noise(img_pil, int(self.adjust_scale))
            img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            print("unavailable adjust_type")

        img = self.transform(img)
        return img, img
        

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

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    
    if(float(adjust_scale) == 1.0):
        nas_set = in_set
        print("ood same as in set")
    else:
        nas_set = Imagenet_adjust_NAS(args.in_datadir, transform = val_tx, adjust_type = adjust_type, adjust_scale = float(adjust_scale))
        print("ood different from in set")
    
    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an ood-distribution set with {len(nas_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)


    nas_loader = torch.utils.data.DataLoader(
        nas_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, nas_set, in_loader, nas_loader


def cal_ood_score(logits, group_slices):
    num_groups = group_slices.shape[0]

    all_group_ood_score_MOS = []

    smax = torch.nn.Softmax(dim=-1).cuda()
    for i in range(num_groups):
        group_logit = logits[:, group_slices[i][0]: group_slices[i][1]]

        group_softmax = smax(group_logit)
        group_others_score = group_softmax[:, 0]

        all_group_ood_score_MOS.append(-group_others_score)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def iterate_data(data_loader, model, group_slices):
    confs_mos = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = model(x)
            conf_mos = cal_ood_score(logits, group_slices)
            confs_mos.extend(conf_mos)

    return np.array(confs_mos)


def run_eval(args, model, in_loader, out_loader, logger, group_slices):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    logger.info("Processing in-distribution data...")
    in_confs = iterate_data(in_loader, model, group_slices)

    logger.info("Processing out-of-distribution data...")
    out_confs = iterate_data(out_loader, model, group_slices)

    in_examples = in_confs.reshape((-1, 1))
    out_examples = out_confs.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    path_auroc_aupr = os.path.join(args.logdir, args.name)
    auc_roc, auc_aupr_in, auc_aupr_out = plot_aupr_auroc(in_examples, out_examples, path_auroc_aupr)

    logger.info('==================== Results for MOS ==================')
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

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    in_set, nas_set, in_loader, nas_loader = mk_id_ood(args, logger, args.adjust_type, args.adjust_scale)

    classes_per_group = np.load(args.group_config)
    args.num_groups = len(classes_per_group)
    group_slices = get_group_slices(classes_per_group)
    group_slices.cuda()
    num_logits = len(in_set.classes) + args.num_groups

    logger.info(f"Loading model from {args.model_path}")
    model = resnetv2.KNOWN_MODELS[args.model](head_size=num_logits)

    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    start_time = time.time()
    run_eval(args, model, in_loader, nas_loader, logger, group_slices)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--group_config", default="group_config/taxonomy_level0.npy")
    parser.add_argument("--adjust_type", required=True)
    parser.add_argument("--adjust_scale", required=True)
    main(parser.parse_args())
