import argparse, os

import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim

from utils.util import create_logger
from  configs.cfg import get_cfg_defaults, update_cfg
from dataio.dataset import get_dataset


    
def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='cnn_backbone') 
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--backbone', type=str, default='')
    
    args.datasets = 'bioscan'

    args = parser.parse_args()

    update_cfg(cfg, args) 

    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'), display=True)

    log(str(cfg))
    
    # Get the dataset for training
    train_loader, train_push_loader, test_loader = get_dataset(cfg, log)

    model = resnet50(pretrained=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.CrossEntropyLoss().to(device)

    num_epochs = 20
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {100 * accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'resnet50_backbone.pth')
 

if __name__ == '__main__':
    main()
    