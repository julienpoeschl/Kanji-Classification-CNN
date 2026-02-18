import torch
import torch.nn as nn
from typing import Optional

from model.src.paths import BEST_MODEL_PATH

class KanjiCNN(nn.Module):
    """
    CNN model for Kanji character classification.
    
    Architecture designed for 64x64 grayscale images:
    - 4 convolutional blocks with batch normalization and dropout
    - Global average pooling
    - Fully connected classifier
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.5) -> None:
        """
        Initialize the Kanji CNN model.
        
        :param num_classes: Number of output classes (kanji characters)
        :param dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Convolutional blocks
        # Input: (N, 1, 64, 64)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (N, 32, 32, 32)
            nn.Dropout2d(0.25)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (N, 64, 16, 16)
            nn.Dropout2d(0.25)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (N, 128, 8, 8)
            nn.Dropout2d(0.25)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (N, 256, 4, 4)
            nn.Dropout2d(0.25)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (N, 256, 1, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param x: Input tensor of shape (N, 1, H, W)
        :return: Output logits of shape (N, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def create_model(
    num_classes: int,
    dropout_rate: float = 0.5,
    device: Optional[torch.device] = None
) -> KanjiCNN:
    """
    Create and return a KanjiCNN model.
    
    :param num_classes: Number of output classes
    :param dropout_rate: Dropout rate for regularization
    :param device: Device to place model on (defaults to CUDA if available)
    :return: Initialized KanjiCNN model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = KanjiCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    return model


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """Create Adam optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 5
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create cosine annealing scheduler with warmup."""
    T_max = max(1, epochs - warmup_epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


def get_criterion() -> nn.Module:
    """Return cross entropy loss for classification."""
    return nn.CrossEntropyLoss()


def load_checkpoint():
    model_path = BEST_MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(model_path, map_location=device)