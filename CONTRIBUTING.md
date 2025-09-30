# Contributing to CIFAR10 Deep Learning Comparison

🎉 Thank you for your interest in contributing to this project! / Bu projeye katkıda bulunmak istediğiniz için teşekkürler!

## How to Contribute / Nasıl Katkıda Bulunabilirsiniz

### 🐛 Reporting Bugs / Hata Bildirimi
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Provide detailed information about the issue
- Include steps to reproduce the problem
- Add relevant screenshots or error logs

### 💡 Suggesting Features / Özellik Önerisi
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Clearly describe the proposed feature
- Explain why it would be beneficial
- Consider implementation complexity

### ❓ Asking Questions / Soru Sorma
- Use the [Question template](.github/ISSUE_TEMPLATE/question.md)
- Search existing issues first
- Provide context and what you've already tried

## Development Setup / Geliştirme Ortamı Kurulumu

### Prerequisites / Ön Koşullar
- Python 3.8 or higher / Python 3.8 veya üzeri
- Git
- CUDA-compatible GPU (optional but recommended) / CUDA uyumlu GPU (opsiyonel ama önerilen)

### Installation / Kurulum
1. Fork the repository / Repository'yi fork edin
2. Clone your fork / Fork'unuzu klonlayın:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CIFAR10-Deep-Learning-Comparison.git
   cd CIFAR10-Deep-Learning-Comparison
   ```

3. Create a virtual environment / Sanal ortam oluşturun:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

4. Install dependencies / Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

5. Download CIFAR10 data / CIFAR10 verisini indirin:
   ```bash
   python download_data.py
   ```

## Code Style / Kod Stili

### Python Code Standards / Python Kod Standartları
- Follow PEP 8 guidelines / PEP 8 kurallarını takip edin
- Use meaningful variable and function names / Anlamlı değişken ve fonksiyon isimleri kullanın
- Add docstrings to functions and classes / Fonksiyon ve sınıflara docstring ekleyin
- Keep functions focused and small / Fonksiyonları odaklı ve küçük tutun

### Example / Örnek:
```python
def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    \"\"\"
    Calculate accuracy between predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float between 0 and 1
    \"\"\"
    correct = (predictions.argmax(dim=1) == targets).float()
    return correct.mean().item()
```

## Pull Request Process / Pull Request Süreci

### Before Submitting / Göndermeden Önce
1. **Test your changes** / Değişikliklerinizi test edin:
   ```bash
   python train.py --epochs 1  # Quick test
   streamlit run app.py  # Test web app
   ```

2. **Update documentation** / Dokümantasyonu güncelleyin:
   - Update README.md if needed / Gerekirse README.md'yi güncelleyin
   - Add comments to complex code / Karmaşık koda yorum ekleyin

3. **Check code quality** / Kod kalitesini kontrol edin:
   ```bash
   # Format code (if you have black installed)
   black *.py
   
   # Check for common issues
   python -m py_compile *.py
   ```

### Submitting / Gönderme
1. Create a new branch / Yeni branch oluşturun:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes / Değişikliklerinizi yapın

3. Commit with clear messages / Açık mesajlarla commit edin:
   ```bash
   git add .
   git commit -m \"Add: New feature description\"
   ```

4. Push to your fork / Fork'unuza push edin:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request / Pull Request oluşturun

### Pull Request Guidelines / Pull Request Kuralları
- **Title**: Use clear, descriptive titles / Açık, tanımlayıcı başlıklar kullanın
- **Description**: Explain what changes you made and why / Ne değiştirdiğinizi ve neden değiştirdiğinizi açıklayın
- **Testing**: Describe how you tested your changes / Değişikliklerinizi nasıl test ettiğinizi açıklayın
- **Screenshots**: Include screenshots for UI changes / UI değişiklikleri için ekran görüntüleri ekleyin

## Types of Contributions / Katkı Türleri

### 🔧 Code Contributions / Kod Katkıları
- Bug fixes / Hata düzeltmeleri
- New features / Yeni özellikler
- Performance improvements / Performans iyileştirmeleri
- Code refactoring / Kod yeniden düzenleme

### 📚 Documentation / Dokümantasyon
- README improvements / README iyileştirmeleri
- Code comments / Kod yorumları
- Tutorial creation / Eğitim materyali oluşturma
- Translation / Çeviri

### 🎨 Design / Tasarım
- UI/UX improvements / UI/UX iyileştirmeleri
- Streamlit app enhancements / Streamlit uygulama geliştirmeleri
- Visualization improvements / Görselleştirme iyileştirmeleri

### 🧪 Testing / Test
- Unit tests / Birim testler
- Integration tests / Entegrasyon testleri
- Performance benchmarks / Performans kıyaslamaları

## Model Contributions / Model Katkıları

### Adding New Models / Yeni Model Ekleme
If you want to add a new model / Yeni bir model eklemek istiyorsanız:

1. **Create model class** / Model sınıfı oluşturun:
   ```python
   class YourModel(nn.Module):
       def __init__(self, num_classes=10):
           # Your implementation
           pass
   ```

2. **Add to training script** / Eğitim scriptine ekleyin
3. **Update Streamlit app** / Streamlit uygulamasını güncelleyin
4. **Test thoroughly** / Kapsamlı test edin
5. **Document performance** / Performansı belgeleyin

### Model Requirements / Model Gereksinimleri
- Must work with CIFAR10 (32x32x3 input) / CIFAR10 ile çalışmalı
- Should achieve reasonable accuracy (>85%) / Makul doğruluk elde etmeli
- Include proper documentation / Uygun dokümantasyon içermeli
- Provide training curves and metrics / Eğitim eğrileri ve metrikleri sağlamalı

## Community Guidelines / Topluluk Kuralları

### Be Respectful / Saygılı Olun
- Use inclusive language / Kapsayıcı dil kullanın
- Be patient with beginners / Yeni başlayanlara sabırlı olun
- Provide constructive feedback / Yapıcı geri bildirim verin

### Be Helpful / Yardımcı Olun
- Answer questions when you can / Yapabildiğinizde soruları yanıtlayın
- Share knowledge and resources / Bilgi ve kaynak paylaşın
- Help review pull requests / Pull request'leri incelemeye yardım edin

## Recognition / Tanınma

Contributors will be recognized in:
Katkıda bulunanlar şuralarda tanınacak:

- README.md contributors section / README.md katkıda bulunanlar bölümü
- Release notes for significant contributions / Önemli katkılar için sürüm notları
- Special thanks in documentation / Dokümantasyonda özel teşekkürler

## Questions? / Sorular?

- Open an issue with the Question template / Soru şablonu ile issue açın
- Check existing issues and discussions / Mevcut issue'ları ve tartışmaları kontrol edin
- Review the README.md for basic information / Temel bilgiler için README.md'yi inceleyin

---

**Happy Contributing! / İyi Katkılar!** 🚀

Thank you for helping make this project better! / Bu projeyi daha iyi hale getirmeye yardım ettiğiniz için teşekkürler!