import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import cv2
from utils import *
from dataset import *
from loss import *
import wandb


# os.environ["CUDA_VISIBLE_DEVICES"]="7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--run_mode', default="test", type=str, help='train or test')
parser.add_argument('--loss', default="ce", type=str, help='ce or ovadm or contrastive or contrastive_mean')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--result_path', default="./results", type=str, help='train or test')
parser.add_argument('--seed', default=0, type=int, help='set seed')
parser.add_argument('--aug_type', default="basic", type=str, help='simclr or basic')
parser.add_argument('--w1', default=1.0, type=float, help='weightage for CE loss')
parser.add_argument('--w2', default=1.0, type=float, help='weightage for MU loss')
parser.add_argument('--w3', default=1.0, type=float, help='weightage for variance loss')
parser.add_argument('--w4', default=1.0, type=float, help='weightage for entropy loss')
parser.set_defaults(argument=True)

features = None
wandb.init(project="ood_experiments_cross-entropy")

def get_features_hook(self, input, output):
    global features
    features = [output]

def get_features(model, data, num_classes):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    handle = model.fc[1].register_forward_hook(get_features_hook)
    out = model(data)
    handle.remove()
    global features
    out_features = features[0]

    return out, out_features


def define_criterion(args, model):
    if (args.loss == 'ce_with_mu_variance'):
        criterion = CEwithContrastive(args, model.fc[2])
    elif(args.loss == 'CEwithMuVarCorr'):
        criterion = CEwithMuVarCorr(args, model.fc[2], args.w2, args.w3, args.w4)
    elif(args.loss == 'ClasswiseHLoss'):
        criterion = ClasswiseHLoss(args, model.fc[2], w1 = args.w2, w2 = args.w4)
    elif(args.loss == 'ClasswiseHLossNeg'):
        criterion = ClasswiseHLossNeg(args, model.fc[2], w1 = args.w2, w2 = args.w4)
    return criterion


def calc_loss(model, criterion, inputs, labels, args):
    loss, loss_ce, loss_mu, loss_var = None, None, None , None
    penulti_ftrs, outputs = None, None
    outputs, penulti_ftrs = get_features(model, inputs, args.num_classes)
    if(args.loss == 'ce_with_mu_variance'):
        loss_ce, loss_mu, loss_var = criterion(penulti_ftrs, labels)
        loss =  args.w1*loss_ce + args.w2*loss_mu + args.w3*loss_var
    elif(args.loss == 'CEwithMuVarCorr') | (args.loss == 'ClasswiseHLoss') | (args.loss == 'ClasswiseHLossNeg'):
        loss  = criterion(penulti_ftrs, labels)
        

    # if args.loss == 'ce':
    #     loss =  args.w1*loss_ce 
    # elif (args.loss == 'ce_with_mu'): 
    #     loss =  args.w1*loss_ce + args.w2*loss_mu
    # elif (args.loss == 'ce_with_variance'):
    #     loss =  args.w1*loss_ce + args.w3*loss_var
    # elif (args.loss == 'ce_with_mu_variance'):
    #     loss =  args.w1*loss_ce + args.w2*loss_mu + args.w3*loss_var
    # elif(args.loss = 'CEwithMuVarCorr'):

    return loss, loss_ce, loss_mu, loss_var, penulti_ftrs, outputs


def prediction(penulti_ftrs, outputs, labels, args, criterion):
    predicted, predicted_value = None, None
    predicted, predicted_value = criterion.predict(penulti_ftrs)

    return predicted, predicted_value

def train(args, bones_df, train_df, val_df, test_df, data_transform):

    ##############################
    # define model/log pth
    ##############################
    model_pth = os.path.join(args.result_path, "models")
    log_pth = os.path.join(args.result_path, "logs")
    os.makedirs(model_pth, exist_ok=True)
    os.makedirs(log_pth, exist_ok=True)

    best_file = os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))
    log_file = os.path.join(args.result_path, "logs", "train_log_{}.txt".format(args.seed))
    with open(log_file, "w") as file:
        file.write("")
    
    #######################################################
    # define dataset , model, criterion & optimizer
    #######################################################
    train_dataset = BoneDataset(dataframe = train_df,img_dir='/home/edlab/radhika/radhika_77/data/datasets/boneage_data_kaggle/boneage-training-dataset/boneage-training-dataset/', mode = 'train', transform = data_transform)
    val_dataset = BoneDataset(dataframe = val_df,img_dir='/home/edlab/radhika/radhika_77/data/datasets/boneage_data_kaggle/boneage-training-dataset/boneage-training-dataset/', mode = 'val', transform = data_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
       
    model = define_model(device)
    model.to(device)
    # model = torch.nn.DataParallel(model)
    criterion = define_criterion(args, model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ##############################
    # training
    ##############################
    iteration_for_summary = 0
    best_acc = 0

    for epoch in range(30):
        model.train()
        running_loss = 0.0
        if(args.loss == 'ce_with_mu') | (args.loss == 'ce_with_variance')| (args.loss =='ce_with_mu_variance') | (args.loss =='ClasswiseHLoss')| (args.loss =='ClasswiseHLossNeg'):
            running_loss_ce, running_loss_mu, running_loss_var =  0.0, 0.0, 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            ########################
            # calc total loss
            ########################
            optimizer.zero_grad()
            loss, loss_ce, loss_mu, loss_var, penulti_ftrs, outputs = calc_loss(model, criterion, inputs, labels, args)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if(args.loss == 'ce_with_mu') | (args.loss == 'ce_with_variance')| (args.loss =='ce_with_mu_variance') | (args.loss =='ClasswiseHLoss'):
                # running_loss_ce += loss_ce.item()
                # running_loss_mu += loss_mu.item()
                # running_loss_var += loss_var.item()
                
            # prediction & acc
            #######################
            predicted, _ = prediction(penulti_ftrs, outputs, labels, args, criterion)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#             if i % 10 == 0:    
        table = '[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss / total)
        train_acc = (100 * correct / total)
        table += 'Train acc: {}'.format(train_acc)

        # print("running_loss, running_loss_ce, running_loss_mu, running_loss_var", running_loss, running_loss_ce, running_loss_mu, running_loss_var)
        # print("running_loss", running_loss)
        with open(log_file, "a") as file:
            file.write(table)

        #######################
        # validation
        #######################
        with torch.no_grad():
            model.eval()
            val_total = 0
            val_correct = 0
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                outputs, penulti_ftrs = get_features(model, inputs, args.num_classes)
                predicted, predicted_value = criterion.predict(penulti_ftrs)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
        val_acc = (100 * val_correct / val_total)
        table = 'Epoch: {}, Validation acc: {}'.format(epoch+1, val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_file)         
            table+= "   <<< best acc"


        print(table)
        with open(log_file, "a") as file:
            file.write(table)
            file.write("\n")
            
        wandb.log({'epoch': epoch, 'loss': running_loss})
        # wandb.log({'epoch': epoch, 'loss_ce': running_loss_ce})
        # wandb.log({'epoch': epoch, 'loss_mu': running_loss_mu})
        # wandb.log({'epoch': epoch, 'loss_var': running_loss_var})
        # wandb.log({'epoch': epoch, 'train_acc': train_acc})
        # wandb.log({'epoch': epoch, 'val_acc': val_acc})

            

def test(args, bones_df, train_df, val_df, test_df, data_transform):        
    log_pth = os.path.join(args.result_path, "logs")
    os.makedirs(log_pth, exist_ok=True)
    log_file = os.path.join(args.result_path, "logs", "test_log_{}.txt".format(args.seed))
    with open(log_file, "w") as file:
        file.write("")
    test_dataset = BoneDataset(dataframe = test_df,img_dir='/home/edlab/radhika/radhika_77/data/datasets/boneage_data_kaggle/boneage-training-dataset/boneage-training-dataset/', mode = 'test', transform = data_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    model = define_model(device)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    criterion = define_criterion(args, model)
    
    model.eval()
    
    with torch.no_grad():
        test_total = 0
        test_correct = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs, penulti_ftrs = get_features(model, inputs, args.num_classes)
            predicted, predicted_value = criterion.predict(penulti_ftrs)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            print(test_total,test_correct)
    table = 'Test acc: {}'.format(100 * test_correct / test_total)
    with open(log_file, "a") as file:
        file.write(table)
        
                

def ood_test(args, bones_df, train_df, val_df, test_df, data_transform, adjust = None):
    if adjust == None:
        age_groups = [[1,2,3,4,5],[6],[7],[8],[9],[10,11,12],[13],[14],[15,16,17,18,19]]
        log_file = os.path.join(args.result_path, "logs", "ood_age_200_test_log_{}.txt".format(args.seed))
        # load dataloaders
        loaders, data_len = get_eval_dataloaders(bones_df, train_df, val_df, test_df, data_transform, age_groups)
        len_ood = len(age_groups)
    else:
        log_file = os.path.join(args.result_path, "logs", "ood_{}_test_log_{}.txt".format(adjust, args.seed))
        loaders, data_len, adjust_scale = get_adjust_dataloaders(bones_df, train_df, val_df, test_df, data_transform, adjust)
    
    with open(log_file, "w") as file:
        file.write("")
    
    # load trained models
    model = define_model(device)
    model.to(device)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    criterion = define_criterion(args, model)
    model.eval()

    # calc Baseline score and classification accuracy
    with torch.no_grad():
        for i, test_loader in enumerate(loaders):
            print('test dataset #{}'.format(str(i+1)))
            test_total = 0
            test_correct = 0
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                outputs, penulti_ftrs = get_features(model, images, args.num_classes)
                predicted, predicted_value = criterion.predict(penulti_ftrs)

                ##### classification accuracy ####
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            if adjust == None:     
                table = 'age: {}, Len:{}, Test acc: {}\n'.format(age_groups[i], data_len[i], (100 * test_correct / test_total))
            else:
                table = 'adjustness: {}, Len:{}, Test acc: {}\n'.format(adjust_scale[i], data_len[i], (100 * test_correct / test_total))
            
            with open(log_file, "a") as file:
                file.write(table)
                
                
def main():
    args = parser.parse_args()
    set_seed(args.seed)
    wandb.run.name = args.result_path + "_" + str(args.seed)
    bones_df, train_df, val_df, test_df, data_transform = Data_Transform()
    if args.run_mode == 'train':
        train(args, bones_df, train_df, val_df, test_df, data_transform)
        test(args, bones_df, train_df, val_df, test_df, data_transform)
    elif args.run_mode == 'test':
        ood_test(args, bones_df, train_df, val_df, test_df, data_transform)
        ood_test(args, bones_df, train_df, val_df, test_df, data_transform, 'bright')
#         ood_test(args, bones_df, train_df, val_df, test_df, data_transform, 'contrast')
#         ood_test(args, bones_df, train_df, val_df, test_df, data_transform, 'impulse_noise')
        # ood_test(args, bones_df, train_df, val_df, test_df, data_transform, 'gaussian_noise')
#         ood_test(args, bones_df, train_df, val_df, test_df, data_transform, 'shot_noise')
    else:
        print("not available mode")

if __name__ == '__main__':
    main()