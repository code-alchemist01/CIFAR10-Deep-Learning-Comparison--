# 🖼️ CIFAR10 Görüntü Sınıflandırma Projesi

Bu proje, CIFAR10 veri seti üzerinde üç farklı derin öğrenme modelini (ResNet18, ResNet34, EfficientNet-B0) eğiterek görüntü sınıflandırma performanslarını karşılaştırmaktadır. Ayrıca, eğitilmiş modelleri test etmek için interaktif bir Streamlit web uygulaması içermektedir.

## 📊 Proje Özeti

- **Veri Seti**: CIFAR10 (10 sınıf, 60.000 görüntü)
- **Modeller**: ResNet18, ResNet34, EfficientNet-B0
- **En İyi Performans**: ResNet34 (%89.61 doğruluk)
- **Web Uygulaması**: Streamlit ile gerçek zamanlı tahmin

## 🎯 CIFAR10 Sınıfları

1. ✈️ **Airplane** (Uçak)
2. 🚗 **Automobile** (Otomobil)
3. 🐦 **Bird** (Kuş)
4. 🐱 **Cat** (Kedi)
5. 🦌 **Deer** (Geyik)
6. 🐕 **Dog** (Köpek)
7. 🐸 **Frog** (Kurbağa)
8. 🐎 **Horse** (At)
9. 🚢 **Ship** (Gemi)
10. 🚛 **Truck** (Kamyon)

## 📈 Model Performansları

| Model | En İyi Doğruluk | Son Doğruluk | Eğitim Süresi |
|-------|----------------|--------------|---------------|
| **ResNet34** | **89.61%** | 89.61% | 86 dk |
| **ResNet18** | **89.24%** | 89.24% | 58 dk |
| **EfficientNet-B0** | **67.71%** | 67.71% | 33 dk |

## 🚀 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Bağımlılıklar

- Python 3.8+
- PyTorch
- Torchvision
- Streamlit
- Plotly
- Pandas
- NumPy
- Pillow
- OpenCV

## 📁 Proje Yapısı

```
Computer_Vision_CIFAR10/
├── app.py                          # Streamlit web uygulaması
├── train.py                        # Model eğitim scripti
├── download_data.py                 # Veri indirme scripti
├── requirements.txt                 # Python bağımlılıkları
├── README.md                        # Bu dosya
├── data/                           # CIFAR10 veri seti
│   ├── cifar-10-batches-py/
│   └── cifar-10-python.tar.gz
├── models/                         # Eğitilmiş model dosyaları
│   ├── resnet18_best.pth
│   ├── resnet34_best.pth
│   └── efficientnet_best.pth
├── plots/                          # Eğitim grafikleri ve confusion matrix
│   ├── resnet18_training_curves.png
│   ├── resnet18_confusion_matrix.png
│   ├── resnet34_training_curves.png
│   ├── resnet34_confusion_matrix.png
│   ├── efficientnet_training_curves.png
│   ├── efficientnet_confusion_matrix.png
│   └── model_comparison.png
└── results/                        # Sonuç dosyaları
    └── model_comparison.csv
```

## 🏃‍♂️ Kullanım

### 1. Veri İndirme

```bash
python download_data.py
```

### 2. Model Eğitimi

```bash
python train.py
```

Bu komut üç modeli sırayla eğitir:
- ResNet18 (30 epoch)
- ResNet34 (30 epoch) 
- EfficientNet-B0 (30 epoch)

### 3. Streamlit Uygulamasını Çalıştırma

```bash
streamlit run app.py
```

Uygulama `http://localhost:8502` adresinde çalışacaktır.

## 🌐 Web Uygulaması Özellikleri

### 🏠 Ana Sayfa
- Proje bilgileri ve model performans metrikleri
- Model karşılaştırma tablosu
- Eğitim süreleri ve doğruluk oranları

### 🔮 Gerçek Zamanlı Tahmin
- Görüntü yükleme arayüzü
- Üç modelden eş zamanlı tahmin
- Güven skorları ile interaktif grafikler
- Tahmin sonuçlarının görselleştirilmesi

  <img width="1880" height="620" alt="Ekran görüntüsü 2025-10-01 014400" src="https://github.com/user-attachments/assets/99a95c5a-dcd2-4cae-904e-49e7cf3b00fb" />
<img width="1876" height="730" alt="Ekran görüntüsü 2025-10-01 014615" src="https://github.com/user-attachments/assets/dd6fa4b0-f894-4372-b7a1-443cfbe85d15" />
<img width="1892" height="733" alt="Ekran görüntüsü 2025-10-01 014601" src="https://github.com/user-attachments/assets/62aee5f6-f224-4a77-b6f5-f190fd88b561" />
<img width="1892" height="758" alt="Ekran görüntüsü 2025-10-01 014756" src="https://github.com/user-attachments/assets/66f34b01-f376-4c4b-bd6a-3d529452d93f" />

### 📊 Model Karşılaştırması
- Detaylı performans analizi
- Eğitim eğrileri görselleştirmesi
- Confusion matrix analizi
- Model özelliklerinin karşılaştırılması

<img width="1875" height="863" alt="Ekran görüntüsü 2025-10-01 014429" src="https://github.com/user-attachments/assets/c3d26f0a-d407-47a7-bff8-59d016b3d443" />

  

## 🧠 Model Mimarileri

### ResNet18 & ResNet34
- **Özel implementasyon** ile sıfırdan eğitim
- Residual bağlantılar ile gradient vanishing problemi çözümü
- CIFAR10 için optimize edilmiş mimari

### EfficientNet-B0
- **Transfer learning** ile ImageNet ağırlıkları
- Compound scaling yöntemi
- Daha az parametre ile yüksek verimlilik

## 📊 Eğitim Detayları

### Hiperparametreler
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 30
- **Scheduler**: StepLR (step_size=10, gamma=0.1)

### Veri Artırma
- RandomHorizontalFlip
- RandomRotation
- Normalizasyon (CIFAR10 standartları)

## 🎨 Görselleştirmeler

Proje şu görselleştirmeleri içerir:
- **Eğitim Eğrileri**: Loss ve accuracy grafikleri
- **Confusion Matrix**: Sınıf bazında performans analizi
- **Model Karşılaştırması**: Performans metrikleri karşılaştırması
- **Gerçek Zamanlı Grafikler**: Streamlit uygulamasında interaktif grafikler

## 🔍 Sonuçlar ve Analiz

### 🏆 En İyi Performans: ResNet34
- **%89.61 doğruluk** ile en yüksek performans
- Daha derin mimari sayesinde daha iyi özellik öğrenme
- Eğitim süresi uzun ancak sonuç tatmin edici

### ⚡ En Hızlı: EfficientNet-B0
- **33.87 dakika** ile en hızlı eğitim
- **4.02M parametre** ile en küçük model
- Transfer learning avantajı

### ⚖️ Denge: ResNet18
- **%89.24 doğruluk** ile iyi performans
- **58.74 dakika** ile orta eğitim süresi
- Performans/süre dengesi açısından optimal

## 🚀 Gelecek Geliştirmeler

- [ ] Data augmentation tekniklerinin genişletilmesi
- [ ] Ensemble yöntemleri ile model kombinasyonu
- [ ] Daha fazla model mimarisi eklenmesi (Vision Transformer, etc.)
- [ ] Hyperparameter tuning ile optimizasyon
- [ ] Model quantization ile deployment optimizasyonu

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 📞 İletişim

Proje hakkında sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
