import argparse, os

import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim

from utils.util import create_logger
from  configs.cfg import get_cfg_defaults
from dataio.dataset import get_dataset


    
def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='cub.yaml')
    args = parser.parse_args()

    # Update the hyperparameters from default to the ones we mentioned in arguments
    cfg.merge_from_file(args.configs)
    
    args.datasets = 'bioscan'

    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        os.mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        os.mkdir(cfg.OUTPUT.IMG_DIR)

    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'), display=True)

    log(str(cfg))
    
    # Get the dataset for training
    train_loader, train_push_loader, test_loader = get_dataset(cfg, log)

    log("Loading model...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.DATASET.NUM_CLASSES)
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=1, threshold=1e-5)
    
    criterion = nn.CrossEntropyLoss().to(device)

    log("Training")

    num_epochs = 20
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
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
        epoch_loss = running_loss / len(train_loader.dataset)
        
        log(f"Testing Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
        log(f"Accuracy on test set: {100 * accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'resnet50_backbone.pth')
 

if __name__ == '__main__':
    main()
    