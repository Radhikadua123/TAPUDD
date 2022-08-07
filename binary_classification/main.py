import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import cv2
import wandb
import time 

from utils import *
from dataset import *

torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--run_mode', default="test", type=str, help='train or test')
# parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--result_path', default="./results", type=str, help='path of model')
parser.add_argument('--seed', default=0, type=int, help='set seed')
parser.add_argument('--variation', default=0, type=str, help='age, bright, contrast, impulse_noise, shot_noise, gaussian_noise')
# parser.add_argument('--aug_type', default="basic", type=str, help='simclr or basic')
parser.add_argument('--epoch', default=30, type=int, help='no of epochs')
parser.add_argument('--data_path', default="./data", type=str, help='path of the dataset')
parser.set_defaults(argument=True)

features = None

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


def train(args, bones_df, train_df, val_df, test_df, data_transform):
    """ Function to train the model."""

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
    images_dir = os.path.join(args.data_path, 'boneage-training-dataset/boneage-training-dataset/')
   
    train_dataset = BoneDataset(dataframe = train_df, img_dir = images_dir, mode = 'train', transform = data_transform)
    val_dataset = BoneDataset(dataframe = val_df, img_dir = images_dir, mode = 'val', transform = data_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
       
    model = define_model(device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ##############################
    # training
    ##############################
    iteration_for_summary = 0
    best_acc = 0
    n_epochs_stop = 20
    epochs_no_improve = 0
    early_stop = True

    for epoch in range(args.epoch):
        start_time = time.process_time()
        model.train()
        running_loss = 0.0
        
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
            #######################   
            # prediction & acc
            #######################
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
#             if i % 10 == 0:    
        table = '[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss / total)
        train_acc = (100 * correct / total)
        table += 'Train acc: {}'.format(train_acc)
        print("epoch: ", epoch, 100 * correct / total)

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
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
        val_acc = (100 * val_correct / val_total)
        table = 'Epoch: {}, Validation acc: {}, epoch time : {} seconds'.format(epoch + 1, val_acc,  time.process_time() - start_time)
        if val_acc >= best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_file)         
            table+= "   <<< best acc"
        else:
            epochs_no_improve += 1

        if epoch > n_epochs_stop and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            print(table)
            with open(log_file, "a") as file:
                file.write(table)
                file.write("\n")
            break

        print(table)
        with open(log_file, "a") as file:
            file.write(table)
            file.write("\n")

            

def test(args, bones_df, train_df, val_df, test_df, data_transform): 
    """ Function to test classification performance of the trained model on in-distribution test data. Calculates accuracy on in-distribution test data."""       
    log_pth = os.path.join(args.result_path, "logs")
    os.makedirs(log_pth, exist_ok=True)
    log_file = os.path.join(args.result_path, "logs", "test_log_{}.txt".format(args.seed))
    with open(log_file, "w") as file:
        file.write("")

    images_dir = os.path.join(args.data_path, 'boneage-training-dataset/boneage-training-dataset/')
    test_dataset = BoneDataset(dataframe = test_df, img_dir = images_dir, mode = 'test', transform = data_transform)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)
    
    model = define_model(device)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    model.eval()
    
    with torch.no_grad():
        test_total = 0
        test_correct = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            print(test_total,test_correct)
    table = 'Test acc: {}'.format(100 * test_correct / test_total)
    with open(log_file, "a") as file:
        file.write(table)
        
                

def ood_test(args, bones_df, train_df, val_df, test_df, data_transform, adjust = "bright"):
    """ Function to test classification performance of the trained model on distributionally shifted datasets. Calculates accuracy on each OOD dataset."""       
    images_dir = os.path.join(args.data_path, 'boneage-training-dataset/boneage-training-dataset/')

    log_file = os.path.join(args.result_path, "logs", "ood_{}_test_log_{}.txt".format(adjust, args.seed))
    loaders, data_len, adjust_scale = get_adjust_dataloaders(bones_df, train_df, val_df, test_df, images_dir, data_transform, adjust)
    
    with open(log_file, "w") as file:
        file.write("")
    
    # Define the model and load the weights of trained model.
    model = define_model(device)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.result_path, "models", "best_{}.pt".format(args.seed))))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    # Calculate classification accuracy.
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
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                ##### classification accuracy ####
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            table = 'adjustness: {}, Len:{}, Test acc: {}\n'.format(adjust_scale[i], data_len[i], (100 * test_correct / test_total))
            
            with open(log_file, "a") as file:
                file.write(table)
                
                
def main():
    args = parser.parse_args()
    set_seed(args.seed)
    # torch.backends.cudnn.benchmark = True

    bones_df, train_df, val_df, test_df, data_transform = Data_Transform(args.data_path)
    if args.run_mode == 'train':
        train(args, bones_df, train_df, val_df, test_df, data_transform)
    elif args.run_mode == 'test':
        test(args, bones_df, train_df, val_df, test_df, data_transform)
    elif args.run_mode == 'ood_test':
        ood_test(args, bones_df, train_df, val_df, test_df, data_transform, args.variation)
    else:
        print("not available mode")

if __name__ == '__main__':
    main()
