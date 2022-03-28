import os
import argparse
import torchvision as tv
import torch
import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", required=True,
                        help="Where to log test info (small).")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", default="BiT-S-R101x1", help="Which variant to use")
    parser.add_argument("--model_path", type=str, help="Path to the finetuned model you want to test")

    return parser


def mk_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr


def plot_aupr_auroc(in_examples, out_examples, path): #### plot fpr tnr curve; prec recall curve; also get aupr, auroc using auc
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))

    fpr, tpr, _ = sk.roc_curve(labels, examples)
    prec, recall, _ = sk.precision_recall_curve(labels, examples)
    thres = num_in/(num_in + num_out)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))

    prec_out, recall_out, _ = sk.precision_recall_curve(labels_rev, examples)
    thres_out = num_out/(num_in + num_out)

    auc_roc = sk.auc(fpr, tpr)
    auc_aupr_in = sk.auc(recall, prec)
    auc_aupr_out = sk.auc(recall_out, prec_out)

    print("auc roc", auc_roc)
    print("auc aupr in", auc_aupr_in)
    print("auc aupr out", auc_aupr_out)

    plt.figure()
    plt.plot(recall, prec, 'm', label="")
    plt.plot([-0.02, 1.02], [thres, thres], 'k--', label="")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title("AUPR (In)")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    path1 = os.path.join(path, 'AUPR_in.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)

    plt.figure()
    plt.plot(recall_out, prec_out, 'm', label="")
    plt.plot([-0.02, 1.02], [thres_out, thres_out], 'k--', label="")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title("AUPR (Out)")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    path1 = os.path.join(path, 'AUPR_out.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)

    plt.figure()
    plt.plot(fpr, tpr, 'm', label="")
    plt.plot([-0.02, 1.02], [-0.02, 1.02], 'k--', label="")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.title("AUROC")
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    path1 = os.path.join(path, 'AUROC.png')
    plt.savefig(path1, bbox_inches='tight', dpi=300)

    return auc_roc, auc_aupr_in, auc_aupr_out
