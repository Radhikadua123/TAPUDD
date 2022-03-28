from utils import log
import resnetv2
import torch
import time

import numpy as np

from utils.test_utils import arg_parser, mk_id_ood, get_measures, plot_aupr_auroc
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score


def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))
            
    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_kl_div(data_loader, model):
    probs, labels = [], []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            prob = m(logits)
            probs.extend(prob.data.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(probs), np.array(labels)


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    # p = np.asarray(p, dtype=np.float)
    # q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)
        print("num output", num_output)
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'KL_Div':
        logger.info("Processing in-distribution data...")
        in_dist_logits, in_labels = iterate_data_kl_div(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_dist_logits, _ = iterate_data_kl_div(out_loader, model)

        class_mean_logits = []
        for c in range(num_classes):
            selected_idx = (in_labels == c)
            selected_logits = in_dist_logits[selected_idx]
            class_mean_logits.append(np.mean(selected_logits, axis=0))
        class_mean_logits = np.array(class_mean_logits)

        logger.info("Compute distance for in-distribution data...")
        in_scores = []
        for i, logit in enumerate(in_dist_logits):
            if i % 100 == 0:
                logger.info('{} samples processed...'.format(i))
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            in_scores.append(-min_div)
        in_scores = np.array(in_scores)

        logger.info("Compute distance for out-of-distribution data...")
        out_scores = []
        for i, logit in enumerate(out_dist_logits):
            if i % 100 == 0:
                logger.info('{} samples processed...'.format(i))
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            out_scores.append(-min_div)
        out_scores = np.array(out_scores)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    dir_path = os.path.join(args.logdir, args.name)
    os.makedirs(dir_path, exist_ok=True)
    file_path_ood_scores = os.path.join(args.logdir, args.name, "ood_scores.npy")
    file_path_id_scores = os.path.join(args.logdir, args.name, "id_scores.npy")

    np.save(file_path_id_scores, in_examples)
    np.save(file_path_ood_scores, out_examples)

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    path_auroc_aupr = os.path.join(args.logdir, args.name)
    auc_roc, auc_aupr_in, auc_aupr_out = plot_aupr_auroc(in_examples, out_examples, path_auroc_aupr)

    logger.info('============Results for {}============'.format(args.score))
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

    torch.backends.cudnn.benchmark = True

    in_set, out_set, in_loader, out_loader = mk_id_ood(args, logger)

    logger.info(f"Loading model from {args.model_path}")
    model = resnetv2.KNOWN_MODELS[args.model](head_size=len(in_set.classes))

    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=len(in_set.classes))
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'KL_Div'], default='MSP')
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    parser.add_argument('--mahalanobis_param_path', help='path to tuned mahalanobis parameters')

    main(parser.parse_args())
