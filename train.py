#!/usr/bin/env python3
"""
CIFAR-10 ResNet Eğitim Scripti
RTX 5060 8GB için optimize edilmiş
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchvision.models import efficientnet_b0, resnet34
import time
from collections import defaultdict

# CIFAR10 Classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create directories for results
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ResNet18 Implementation
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ResNet34 Implementation
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# EfficientNet Wrapper
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Advanced Data Augmentation
class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_batch, targets_a, targets_b, lam

def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)

def get_data_loaders(batch_size=128, use_advanced_augmentation=True):
    # Advanced augmentation transforms
    if use_advanced_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader

def train_model(model_name='resnet18', epochs=50, batch_size=128, learning_rate=0.1, use_mixup=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model selection
    if model_name == 'resnet18':
        model = ResNet18(num_classes=10)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=10)
    elif model_name == 'efficientnet':
        model = EfficientNetB0(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35], gamma=0.1)
    
    # MixUp
    mixup = MixUp(alpha=1.0) if use_mixup else None
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\nTraining {model_name.upper()} model...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_mixup and mixup:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a).sum().float() + 
                           (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Testing
        test_acc = test_model(model, test_loader, device, verbose=False)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
    
    print(f'\nBest Test Accuracy: {best_test_acc:.2f}%')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, test_accuracies, model_name)
    
    return model, best_test_acc, (train_losses, train_accuracies, test_accuracies)

def test_model(model, test_loader, device, verbose=True):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    if verbose:
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, target_names=CIFAR10_CLASSES))
    
    return accuracy

def plot_confusion_matrix(model, test_loader, device, model_name):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES)
    plt.title(f'{model_name.upper()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses, train_accuracies, test_accuracies, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title(f'{model_name.upper()} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(test_accuracies, label='Test Accuracy', color='red')
    ax2.set_title(f'{model_name.upper()} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_models():
    """Compare multiple models and create comparison plots"""
    models_to_compare = ['resnet18', 'resnet34', 'efficientnet']
    results = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_data_loaders(batch_size=128)
    
    print("🚀 Starting Model Comparison...")
    
    for model_name in models_to_compare:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        start_time = time.time()
        model, best_acc, history = train_model(model_name, epochs=30, use_mixup=True)
        training_time = time.time() - start_time
        
        # Test final model
        model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        final_acc = test_model(model, test_loader, device, verbose=True)
        
        # Generate confusion matrix
        plot_confusion_matrix(model, test_loader, device, model_name)
        
        results[model_name] = {
            'best_accuracy': best_acc,
            'final_accuracy': final_acc,
            'training_time': training_time,
            'history': history
        }
        
        print(f"✅ {model_name.upper()} completed - Best Acc: {best_acc:.2f}%")
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Save results
    import pandas as pd
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Best_Accuracy': [results[m]['best_accuracy'] for m in results.keys()],
        'Final_Accuracy': [results[m]['final_accuracy'] for m in results.keys()],
        'Training_Time_Minutes': [results[m]['training_time']/60 for m in results.keys()]
    })
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    print("\n🎉 Model Comparison Complete!")
    print(comparison_df)
    
    return results

def create_comparison_plots(results):
    """Create comprehensive comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(results.keys())
    colors = ['blue', 'red', 'green']
    
    # 1. Accuracy Comparison
    accuracies = [results[m]['best_accuracy'] for m in models]
    bars1 = ax1.bar(models, accuracies, color=colors)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. Training Time Comparison
    times = [results[m]['training_time']/60 for m in models]
    bars2 = ax2.bar(models, times, color=colors)
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Time (minutes)')
    for bar, time in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time:.1f}m', ha='center', va='bottom')
    
    # 3. Training Curves - Loss
    for i, model in enumerate(models):
        train_losses = results[model]['history'][0]
        ax3.plot(train_losses, label=f'{model.upper()}', color=colors[i])
    ax3.set_title('Training Loss Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Training Curves - Accuracy
    for i, model in enumerate(models):
        test_accs = results[model]['history'][2]
        ax4.plot(test_accs, label=f'{model.upper()}', color=colors[i])
    ax4.set_title('Test Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run comprehensive model comparison
    results = compare_models()
    
    print("\n🎯 CIFAR10 Enhanced Training Complete!")
    print("📊 Check 'plots/' folder for visualizations")
    print("💾 Check 'models/' folder for saved models")
    print("📈 Check 'results/' folder for comparison data")