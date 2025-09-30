#!/usr/bin/env python3
"""
CIFAR-10 Dataset İndirme Scripti
GPU: RTX 5060 8GB için optimize edilmiş
"""

import os
import kaggle
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def download_cifar10():
    """CIFAR-10 datasetini indir ve hazırla"""
    
    # Veri klasörünü oluştur
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🔄 CIFAR-10 dataset indiriliyor...")
    
    # PyTorch'un built-in CIFAR-10 datasetini kullan
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Training set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # Test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    print(f"✅ CIFAR-10 dataset hazır!")
    print(f"📊 Training samples: {len(trainset)}")
    print(f"📊 Test samples: {len(testset)}")
    print(f"📊 Classes: {trainset.classes}")
    
    return trainset, testset

def check_gpu():
    """GPU durumunu kontrol et"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  GPU bulunamadı, CPU kullanılacak")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 CIFAR-10 Computer Vision Projesi")
    print("=" * 50)
    
    # GPU kontrolü
    gpu_available = check_gpu()
    
    # Dataset indir
    trainset, testset = download_cifar10()
    
    print("\n🎉 Hazırlık tamamlandı!")
    print("▶️  Şimdi 'python train.py' komutunu çalıştırabilirsiniz")