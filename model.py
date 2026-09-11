import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Einfaches Convolutional Neural Network
    für die CIFAR-10 Bildklassifikation.
    """

    def __init__(self):
        super().__init__()

        # Erste Convolution-Schicht
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # Zweite Convolution-Schicht
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max-Pooling (Halbierung der Bildgröße)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        # Aktivierungsfunktion
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Vorwärtsdurchlauf des Netzwerks.
        """

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Feature Maps flatten
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        return self.fc2(x)
