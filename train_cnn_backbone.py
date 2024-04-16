import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger

from  configs.cfg import get_cfg_defaults, update_cfg
from dataio.dataset import get_dataset
from augmentation.img_preprocess import preprocess_cub_input_function

from model.model import construct_ppnet
from model.utils import get_optimizers
import train.train_and_test as tnt

from model.model import GeneticCNN2D

import prototype.push as push

    
def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='0') 
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--backbone', type=str, default='')

    args = parser.parse_args()
    args.dataset= "genetics"

    update_cfg(cfg, args) 

    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'), display=True)

    log(str(cfg))
    
    # Get the dataset for training
    train_loader, train_push_loader, validation_loader = get_dataset(cfg, log)

    classes, sizes = train_loader.dataset.get_classes(cfg.DATASET.BIOSCAN.TAXONOMY_NAME)

    model = GeneticCNN2D(720, len(classes), include_connected_layer=True).cuda() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = 1 / torch.tensor(sizes, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    for epoch in range(7):
        running_loss = 0.0
        correct_guesses = 0
        total_guesses = 0
        model.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_pred = torch.argmax(outputs, dim=1)
            correct_guesses += torch.sum(y_pred == labels)
            total_guesses += len(y_pred)

            if i % 10 == 0:
                log(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} accuracy: {correct_guesses / total_guesses}")
                running_loss = 0.0
        
        # Evaluate on test set with balanced accuracy
        model.eval()
        correct_guesses = [0 for _ in range(len(classes))]
        total_guesses = [0 for _ in range(len(classes))]

        with torch.no_grad():
            for data in validation_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                y_pred = torch.argmax(outputs, dim=1)

                for i in range(len(classes)):
                    correct_guesses[i] += torch.sum((y_pred == labels) & (labels == i))
                    total_guesses[i] += torch.sum(labels == i)
        
        accuracy = [correct_guesses[i] / max(1, total_guesses[i]) for i in range(len(classes))]
        balanced_accuracy = sum(accuracy) / len(classes)
        log(f"Epoch {epoch + 1} balanced accuracy: {balanced_accuracy}")

        # Save the model
        if epoch >= 2:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"{args.name}_{epoch}.pth"))
        

if __name__ == '__main__':
    main()
    