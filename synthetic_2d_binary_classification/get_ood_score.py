import pickle
import argparse
import tqdm

import torch
import torch.nn as nn
import os

from utils import define_model, get_model_pth, get_result_pth
from dataset import get_dataloaders, get_planeloader

from odin import sample_odin_estimator, get_odin_score
from mahalanobis import get_feature_list, sample_estimator, get_Mahalanobis_score

features = None


def test(model, args, ood_loader = None, odin_loader=None, mahala_train_loader=None):
    dict_results = dict()
    dict_results['data'] = []
    dict_results['score'] = []

    if 'mahalanobis' in args.metric:
        feature_list = get_feature_list(model, args, device=args.device)
        mean, var = sample_estimator(model, args, num_classes=args.num_classes, feature_list=feature_list,
                                     train_loader=mahala_train_loader,
                                     device=args.device)

    if 'odin' in args.metric:
        odin_criterion = nn.CrossEntropyLoss()
        epsilon = sample_odin_estimator(model, odin_criterion, odin_loader,
                                        epsilons=[0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])



    for idx, data in enumerate(ood_loader):
        inputs, labels = data
        inputs = inputs.to(args.device)

        outputs, penulti_ftrs = model(inputs)
        outputs = torch.nn.Softmax(dim=1)(outputs)
        predicted_value, predicted = torch.max(outputs.data, 1)

        # get ood statistics
        if args.metric == 'baseline':
            score = predicted_value

        elif 'mahalanobis' in args.metric:
            score, _ = get_Mahalanobis_score(model, args, inputs, num_classes=args.num_classes,
                                                     sample_mean=mean, precision=var)

        elif 'odin' in args.metric:
            score = get_odin_score(args, model, inputs, temperature=1000, epsilon=epsilon,
                                           criterion=odin_criterion)
        else:
            print('metric is not available')

        dict_results['data'] += inputs.tolist()
        dict_results['score'] += score.tolist()

    if args.flag_indist:
        file_path = os.path.join(args.result_path, 'scores(train).pkl')
    else:
        file_path = os.path.join(args.result_path, 'scores.pkl')
    with open(file_path, "wb") as f:
        pickle.dump(dict_results, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='type of data')
    parser.add_argument('--root_path', type=str, default='./results', help='root path')
    parser.add_argument('--loss', type=str, default='ce', help='loss')
    parser.add_argument('--metric', type=str, default='baseline', help='score metric')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--flag_indist', action='store_true', help='indistribution dataset')
    parser.add_argument('--input_dim', default=2, type=int, help='seed')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.result_path = get_result_pth(args)

    if args.data == 'circles':
        args.num_classes = 10
    elif args.data == 'flowers':
        args.num_classes = 8
    elif args.data == 'multicircle':
        args.num_classes = 8
    elif args.data == 'multioval':
        args.num_classes = 8
    elif args.data == 'multicircle_10dim':
        args.num_classes = 8
    elif args.data == 'multioval_10dim':
        args.num_classes = 8
    else:
        args.num_classes = 2

    print(args)

    ##############################
    # Data
    ##############################
    loaders = get_dataloaders(args.data)
    train_loader = loaders['train']
    val_loader = loaders['val']

    plane_loader = get_planeloader(args.input_dim)

    ##############################
    # Model
    ##############################
    model = define_model(input_size=args.input_dim, hidden_size = 10, num_classes=args.num_classes)
    model = model.to(args.device)
    model_path = get_model_pth(args)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    if args.flag_indist:
        test(model, args, ood_loader=train_loader, odin_loader=val_loader, mahala_train_loader=train_loader)
    else:
        test(model, args, ood_loader=plane_loader, odin_loader=val_loader,mahala_train_loader=train_loader)

if __name__ == '__main__':
    main()