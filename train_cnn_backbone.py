import argparse, os
import torch
from dataio.genetics import GeneticDataset
from utils.util import save_model_w_condition, create_logger
from torch.utils.data import DataLoader

from dataio.dataset import get_dataset
from augmentation.img_preprocess import preprocess_cub_input_function

from model.model import construct_ppnet
from model.utils import get_optimizers
import train.train_and_test as tnt

from model.model import GeneticCNN2D

import prototype.push as push

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help="Path to training data")
    parser.add_argument('validate', type=str, help="Path to validation data")
    parser.add_argument('--name', type=str, default='cnn_backbone') 
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--output', type=str, default='backbone_temp')
    parser.add_argument('--taxonomy', type=str, default='family')

    args = parser.parse_args()
    args.dataset= "genetics"

    if not os.path.exists(os.path.join(args.output, args.name)):
        os.mkdir(os.path.join(args.output, args.name))

    # Create Logger Initially
    log, logclose = create_logger(log_filename=os.path.join(args.output, args.name, 'train.log'), display=True)

    # Get the dataset for training
    train_dataset = GeneticDataset(args.train,
                            "onehot", 
                            args.taxonomy)
    
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=False)
    
    classes, sizes = train_dataset.get_classes(args.taxonomy)
    
    validation_dataset = GeneticDataset(args.validate, 
                            "onehot",
                            args.taxonomy,
                            classes)
    
    validation_loader = DataLoader(
        validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=False)

    log(f"Training Samples:\t{len(train_dataset)}")
    log(f"Validation Samples:\t{len(validation_dataset)}")
    log(f"Training Classes:\t{len(classes)}")
    validation_classes, validation_sizes = validation_dataset.get_classes(args.taxonomy)
    log(f"Validation Classes:\t{len(validation_classes)}")
    log(f"Class Sizes:")
    for c, s in zip(classes, sizes):
        if c in validation_classes:
            log(f"\t{c + ':':<20}\t{s}\t{validation_sizes[validation_classes.index(c)]}")
        else:
            log(f"\t{c + ':':<20}\t{s}\t0")

    model = GeneticCNN2D(720, len(classes), include_connected_layer=True).cuda() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = 1 / torch.tensor(sizes, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    max_balanced_accuracy = 0
    max_balanced_accuracy_epoch = 0

    for epoch in range(8):
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
        balanced_accuracy = sum(accuracy) / len(validation_classes)
        log(f"Epoch {epoch + 1} balanced accuracy: {balanced_accuracy}")

        if balanced_accuracy > max_balanced_accuracy:
            max_balanced_accuracy = balanced_accuracy
            max_balanced_accuracy_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output, args.name, f"{args.name}_best.pth"))

        # Save the model
        if epoch >= 2 and balanced_accuracy > max_balanced_accuracy:
            torch.save(model.state_dict(), os.path.join(args.output, args.name, f"{args.name}_{epoch}.pth"))
        
    print(f"Best Balanced Accuracy: {max_balanced_accuracy:.4f} at epoch {max_balanced_accuracy_epoch+1}")

if __name__ == '__main__':
    main()
    