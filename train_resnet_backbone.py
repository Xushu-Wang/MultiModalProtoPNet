import argparse, os

import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim

from utils.util import create_logger
from  configs.cfg import get_cfg_defaults
from dataio.dataset import get_dataset

class ComboModel(nn.Module):
    def __init__(self, resnet, output_channels=128, logit_count=40):
        super(ComboModel, self).__init__()
        self.resnet = resnet
        self.channel_fixer = nn.Conv2d(2048,output_channels, kernel_size=(1,1))
        self.fc = nn.Linear(output_channels*64, logit_count)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.channel_fixer(x)
        x = torch.flatten(x,1)
        # x = self.fc(x)
        return x
    
class ComboFCL(nn.Module):
    def __init__(self, in_channels, output_channels=128, logit_count=40):
        super(ComboFCL, self).__init__()
        self.channel_fixer = nn.Conv1d(in_channels,output_channels, kernel_size=(1,1))
        self.fc = nn.Linear(output_channels, logit_count)
        
    def forward(self, x):
        x = self.channel_fixer(x)
        x = self.fc(x)
        return x

    
def main():
    cfg = get_cfg_defaults()

    print(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='cub.yaml')
    args = parser.parse_args()

    # Update the hyperparameters from default to the ones we mentioned in arguments
    cfg.merge_from_file(args.configs)
    
    print(cfg)

    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        os.mkdir(cfg.OUTPUT.MODEL_DIR)


    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'), display=True)

    log(str(cfg))
    
    # Get the dataset for training
    train_loader, train_push_loader, test_loader = get_dataset(cfg, log)

    log("Loading model... \n")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # model = resnet50(weights='DEFAULT')
    # num_ftrs = model.fc.in_features
    # model.fc = ComboFCL(num_ftrs, logit_count=cfg.DATASET.NUM_CLASSES)

    resnet_model = resnet50(weights='DEFAULT')
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-2]))
    model = ComboModel(resnet_model, output_channels=128, logit_count=cfg.DATASET.NUM_CLASSES)
    
    my_dict = torch.load("resnet50_backbone_final_small.pth")
    model.load_state_dict(my_dict)

    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=1, threshold=1e-6)
    
    criterion = nn.CrossEntropyLoss().to(device)

    log("Training \n")

    num_epochs = 60
    
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            print(outputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            training_loss += loss.item() * inputs.size(0)
            
        epoch_loss = training_loss / len(train_loader.dataset)
        
        scheduler.step(epoch_loss)
        
        log(f"Training Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

        # Evaluate the model
        model.eval()
        correct = 0
        total = 0
        
        testing_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = criterion(outputs, labels)
                testing_loss += loss.item() * inputs.size(0)

        accuracy = correct / total
        epoch_loss = testing_loss / len(test_loader.dataset)
        
        log(f"Testing Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
        log(f"Accuracy on test set: {100 * accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'resnet50_backbone_final_small.pth')
 

if __name__ == '__main__':
    main()
    