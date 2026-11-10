import torch
import torch.nn as nn
import torch.nn.functional as F

# fixed constant - see original paper
DECAY_RATE = 1e-6

class CNN_for_CIFAR(nn.Module) :

    """
        basic convolutional neural network from 
        https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#
    """

    def __init__(self) :

        super().__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        #           padding=0, dilation=1, groups=1, bias=True, 
        #           padding_mode='zeros', device=None, dtype=None)

        # torch.nn.Dropout(p=0.5, inplace=False)

        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, 
        #           dilation=1, return_indices=False, ceil_mode=False)

        # padding P = (K - 1) / 2 with K kernel size

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, padding = 2, stride = 1) # 3 input channels because RGB images
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, padding = 2, stride = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1)

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.drop_25 = nn.Dropout2d(p = 0.25)
        self.drop_50 = nn.Dropout2d(p = 0.50)

        # 1024 bc 64 * 64 / 4
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x) :
        
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.drop_25(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.drop_25(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.drop_25(x)

        x = torch.flatten(x, 1) # not squashing the batch dimension
        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop_50(x)
        x = self.fc2(x)
        # x = F.softmax(x) softmax is included with the cross-entropy loss

        return x
