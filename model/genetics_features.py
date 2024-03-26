import torch.nn as nn
import torch
import torch.nn.functional as F

class GeneticCNN2D(nn.Module):
    """Takes a (4, length) tensor and returns a (class_count,) tensor.

    Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
    """

    # Drop last layer and load weights into Proto Layer

    def __init__(self, length:int, class_count:int, include_connected_layer:bool):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1,3), padding=(0,1))
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,3), stride=3)

        if include_connected_layer:
            self.fc1 = nn.Linear(128 * (length // 8), class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        if hasattr(self, 'fc1'):
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            # return F.log_softmax(x, dim=1)
        return x
    

class GeneticCNN(nn.Module):
    """Takes a (4, length) tensor and returns a (class_count,) tensor.

    Layers were chosen arbitrarily, and should be optimized. I have no idea what I'm doing.
    """

    def __init__(self, length:int, class_count:int, two_dimensional:bool=False):
        super().__init__()

        if two_dimensional:
            self.conv1 = nn.Conv2d(4, 32, kernel_size=(1,3), padding=(0,1))
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
            self.conv3 = nn.Conv2d(64, 128, kernel_size=(1,3), padding=(0,1))
            self.pool =  nn.MaxPool2d(kernel_size=(1,2), stride=2)
        else:
            self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(128 * (length // 8), 512)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        return x
    