import torch
import torch.nn as nn
import torch.optim as optim
import argparse


from utils import define_model, set_seed, get_model_pth, get_log_pth
from dataset import get_dataloaders
from losses import HLoss, MULoss, SVD_L
    
def train(args, model, model_path, epochs = 100, flag_mu = False, flag_entropy = False, flag_var=False, loaders = None, device = None):
    train_loader = loaders['train']
    valid_loader = loaders['val']
        
    patience = 10
    
    criterion = nn.CrossEntropyLoss()
    if flag_mu:
        criterion_mu = MULoss(args)
    if flag_entropy:
        criterion_entropy = HLoss()
    if flag_var:
        criterion_var = SVD_L()
#     optimizer = optim.Adam(model.parameters(), lr=0.0006) # ce
    optimizer = optim.Adam(model.parameters(), lr=0.0006)
      
    #training
    max_acc = None
    for epoch in range(epochs):
        # total = 0
        # correct = 0
        model.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, pen_features = model(inputs)

            loss = criterion(outputs, labels)

            if flag_mu:
                loss_mu = criterion_mu(pen_features, labels)
                loss += loss_mu * 0.006
            if flag_entropy:
                loss_entropy = criterion_entropy(pen_features)
                loss += loss_entropy *  0.000002
            if flag_var:
                loss_var = criterion_var(pen_features)
                loss += loss_var * 0.000006
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss)
            
            # predicted_value, predicted = torch.max(outputs.data, 1)
            
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)    
                val_loss += loss.item()

                predicted_value, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(correct)
            print(total)
            acc = 100 * correct / total
            val_loss = val_loss / len(valid_loader)

            print('Epoch [{}/{}]: Accuracy: {}, Loss: {} '.format(epoch+1, epochs,
                        acc, val_loss))

        if max_acc is None:
            max_acc = acc
        if (acc >= max_acc):
            print("saved!")
            patience = 10
            max_acc = acc
            torch.save(model.state_dict(), model_path)
        else:
            patience -= 1
        if(patience == 0):
            break
            
def test(model, test_loader, args, device):
    criterion = nn.CrossEntropyLoss()

    log_file = get_log_pth(args)
    with open(log_file, "w") as file:
        file.write("")
        
    with torch.no_grad():
        model.eval()
        val_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)    
            val_loss += loss.item()

            predicted_value, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

    table = 'Test acc: {}'.format(test_acc)
    with open(log_file, "a") as file:
        file.write(table)
    
            
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of the model')
    parser.add_argument('--root_path', type=str, default='./results', help='path of the model')
    parser.add_argument('--loss', type=str, default='ce', help='path of the model')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--input_dim', default=2, type=int, help='seed')
    parser.add_argument('--flag_mu', action='store_true', help='get margin or not')
    parser.add_argument('--flag_entropy', action='store_true', help='get margin or not')
    parser.add_argument('--flag_var', action='store_true', help='get margin or not')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    model_path = get_model_pth(args)
        
    ############################
    # Data
    ############################
    loaders = get_dataloaders(args.data)

    ##############################
    # Model
    ##############################
    model = define_model(input_size=args.input_dim, hidden_size = 10, num_classes=args.num_classes)
        
    model = model.to(device)
    model.train()
    
    ###############################
    # Training
    ###############################
    train(args, model, model_path, epochs=100, flag_mu=args.flag_mu, flag_entropy=args.flag_entropy, flag_var=args.flag_var, loaders=loaders, device=device)

    model = define_model(input_size=args.input_dim, hidden_size = 10, num_classes=args.num_classes)
    model = model.to(device)
    model_path = get_model_pth(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test(model, loaders['test'], args, device)
           
if __name__ == '__main__':
    main()
