"""
Model module for cat/dog image classification.
Contains LeNet, custom CNN, and other model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any


class LeNet(nn.Module):
    """LeNet-5 architecture adapted for cat/dog classification."""

    def __init__(self, num_classes: int = 2):
        super(LeNet, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Input: 3x224x224, Output: 6x224x224
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 6x112x112
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)           # Output: 16x108x108
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 16x54x54

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, 16 * 54 * 54)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class CustomCNN(nn.Module):
    """Custom CNN architecture for cat/dog classification."""

    def __init__(self, num_classes: int = 2):
        super(CustomCNN, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNetWrapper(nn.Module):
    """Wrapper for ResNet pretrained models."""

    def __init__(self, model_name: str = 'resnet18', num_classes: int = 2, pretrained: bool = True):
        super(ResNetWrapper, self).__init__()

        # Load pretrained ResNet
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze early layers
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetWrapper(nn.Module):
    """Wrapper for EfficientNet pretrained models."""

    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetWrapper, self).__init__()

        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == 'efficientnet_b2':
            base_model = models.efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze early layers
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace the classifier
        num_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ModelFactory:
    """Factory class to create different model architectures."""

    @staticmethod
    def create_model(model_name: str, **kwargs) -> nn.Module:
        """
        Create a model by name.

        Args:
            model_name: Name of the model ('lenet', 'custom_cnn', 'resnet18', 'efficientnet_b0', etc.)
            **kwargs: Additional arguments for the model

        Returns:
            PyTorch model
        """
        model_name = model_name.lower()

        if model_name == 'lenet':
            return LeNet(**kwargs)
        elif model_name == 'custom_cnn':
            return CustomCNN(**kwargs)
        elif model_name.startswith('resnet'):
            return ResNetWrapper(model_name=model_name, **kwargs)
        elif model_name.startswith('efficientnet'):
            return EfficientNetWrapper(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_available_models() -> list:
        """Get list of available model names."""
        return ['lenet', 'custom_cnn', 'resnet18', 'resnet34', 'resnet50',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_name = model_name.lower()

        info = {
            'lenet': {
                'description': 'LeNet-5 architecture, lightweight',
                'parameters': '~60K',
                'recommended_for': 'Quick experiments, educational purposes'
            },
            'custom_cnn': {
                'description': 'Custom CNN with batch normalization',
                'parameters': '~1.2M',
                'recommended_for': 'Balanced performance and speed'
            },
            'resnet18': {
                'description': 'ResNet-18 pretrained on ImageNet',
                'parameters': '~11M',
                'recommended_for': 'Good accuracy with transfer learning'
            },
            'efficientnet_b0': {
                'description': 'EfficientNet-B0 pretrained on ImageNet',
                'parameters': '~5M',
                'recommended_for': 'Best accuracy with efficient computation'
            }
        }

        # Get base name for resnet and efficientnet variants
        base_name = model_name
        if model_name.startswith('resnet'):
            base_name = 'resnet18' if '18' in model_name else 'resnet18'
        elif model_name.startswith('efficientnet'):
            base_name = 'efficientnet_b0'

        return info.get(base_name, {'description': 'Unknown model'})


def test_models():
    """Test different model architectures."""
    print("Testing model architectures...")

    # Create test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)

    # Test each model
    model_names = ModelFactory.get_available_models()

    for model_name in model_names[:3]:  # Test first 3 models
        try:
            print(f"\nTesting {model_name}:")
            model = ModelFactory.create_model(model_name, num_classes=2)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

            # Test forward pass
            with torch.no_grad():
                output = model(test_input)
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_models()
