import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# CIFAR10 sınıf isimleri
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Türkçe sınıf isimleri
CIFAR10_CLASSES_TR = [
    'uçak', 'otomobil', 'kuş', 'kedi', 'geyik',
    'köpek', 'kurbağa', 'at', 'gemi', 'kamyon'
]

# ResNet Block Implementation (same as training)
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

# ResNet18 Implementation (same as training)
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

# ResNet34 Implementation (same as training)
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

# EfficientNet Implementation (same as training)
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

@st.cache_resource
def load_models():
    """Eğitilmiş modelleri yükle"""
    models_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model yolları
    model_paths = {
        'ResNet18': 'models/resnet18_best.pth',
        'ResNet34': 'models/resnet34_best.pth',
        'EfficientNet-B0': 'models/efficientnet_best.pth'
    }
    
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            try:
                if model_name == 'ResNet18':
                    model = ResNet18(num_classes=10)
                elif model_name == 'ResNet34':
                    model = ResNet34(num_classes=10)
                elif model_name == 'EfficientNet-B0':
                    model = EfficientNetB0(num_classes=10)
                
                # Model ağırlıklarını yükle
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)
                model.eval()
                
                models_dict[model_name] = model
                st.success(f"✅ {model_name} başarıyla yüklendi!")
            except Exception as e:
                st.error(f"❌ {model_name} yüklenirken hata: {str(e)}")
        else:
            st.warning(f"⚠️ {model_name} model dosyası bulunamadı: {path}")
    
    return models_dict, device

@st.cache_data
def load_model_results():
    """Model karşılaştırma sonuçlarını yükle"""
    results_path = 'results/model_comparison.csv'
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None

def get_image_transforms():
    """Görüntü ön işleme transformasyonları"""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def predict_image(image, models_dict, device):
    """Görüntüyü sınıflandır"""
    transform = get_image_transforms()
    
    # Görüntüyü RGB'ye çevir
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transformasyon uygula
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    predictions = {}
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # En yüksek olasılıklı sınıfı bul
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
            
            predictions[model_name] = {
                'class_idx': predicted_class_idx,
                'class_name': CIFAR10_CLASSES[predicted_class_idx],
                'class_name_tr': CIFAR10_CLASSES_TR[predicted_class_idx],
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()
            }
    
    return predictions

def create_prediction_chart(predictions):
    """Tahmin sonuçları için grafik oluştur"""
    fig = make_subplots(
        rows=1, cols=len(predictions),
        subplot_titles=list(predictions.keys()),
        specs=[[{"type": "bar"}] * len(predictions)]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        fig.add_trace(
            go.Bar(
                x=CIFAR10_CLASSES_TR,
                y=pred['probabilities'],
                name=model_name,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # En yüksek tahmin için vurgu
        max_idx = pred['class_idx']
        fig.add_trace(
            go.Bar(
                x=[CIFAR10_CLASSES_TR[max_idx]],
                y=[pred['probabilities'][max_idx]],
                name=f"{model_name} - En Yüksek",
                marker_color='gold',
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Model Tahminleri Karşılaştırması",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Olasılık", range=[0, 1])
    
    return fig

def create_model_comparison_chart(results_df):
    """Model performans karşılaştırma grafiği"""
    if results_df is None:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Doğruluk Oranları', 'Eğitim Süreleri'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Doğruluk oranları
    fig.add_trace(
        go.Bar(
            x=results_df['Model'],
            y=results_df['Best_Accuracy'],
            name='Doğruluk (%)',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{acc:.2f}%" for acc in results_df['Best_Accuracy']],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Eğitim süreleri
    fig.add_trace(
        go.Bar(
            x=results_df['Model'],
            y=results_df['Training_Time_Minutes'],
            name='Süre (dk)',
            marker_color=['#FFB6C1', '#98FB98', '#87CEEB'],
            text=[f"{time:.1f} dk" for time in results_df['Training_Time_Minutes']],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Model Performans Karşılaştırması",
        height=400,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Doğruluk (%)", row=1, col=1)
    fig.update_yaxes(title_text="Eğitim Süresi (dk)", row=1, col=2)
    
    return fig

def main():
    st.set_page_config(
        page_title="CIFAR10 Görüntü Sınıflandırma",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ CIFAR10 Görüntü Sınıflandırma Uygulaması")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigasyon")
    page = st.sidebar.selectbox(
        "Sayfa Seçin:",
        ["🏠 Ana Sayfa", "🔍 Görüntü Sınıflandırma", "📈 Model Karşılaştırması"]
    )
    
    if page == "🏠 Ana Sayfa":
        st.header("Hoş Geldiniz!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Proje Hakkında")
            st.write("""
            Bu uygulama, CIFAR10 veri seti üzerinde eğitilmiş üç farklı derin öğrenme modelini kullanarak 
            görüntü sınıflandırması yapar:
            
            - **ResNet18**: 11.17M parametre
            - **ResNet34**: 21.28M parametre  
            - **EfficientNet-B0**: 4.02M parametre
            
            Uygulama, yüklediğiniz görüntüleri 10 farklı sınıfa ayırabilir.
            """)
        
        with col2:
            st.subheader("🎯 CIFAR10 Sınıfları")
            classes_df = pd.DataFrame({
                'İngilizce': CIFAR10_CLASSES,
                'Türkçe': CIFAR10_CLASSES_TR
            })
            st.dataframe(classes_df, use_container_width=True)
        
        # Model sonuçlarını göster
        results_df = load_model_results()
        if results_df is not None:
            st.subheader("🏆 Model Performansları")
            
            # En iyi modeli vurgula
            best_model_idx = results_df['Best_Accuracy'].idxmax()
            best_model = results_df.loc[best_model_idx, 'Model']
            best_accuracy = results_df.loc[best_model_idx, 'Best_Accuracy']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🥇 En İyi Model", 
                    best_model, 
                    f"{best_accuracy:.2f}%"
                )
            
            with col2:
                fastest_idx = results_df['Training_Time_Minutes'].idxmin()
                fastest_model = results_df.loc[fastest_idx, 'Model']
                fastest_time = results_df.loc[fastest_idx, 'Training_Time_Minutes']
                st.metric(
                    "⚡ En Hızlı Eğitim", 
                    fastest_model, 
                    f"{fastest_time:.1f} dk"
                )
            
            with col3:
                total_time = results_df['Training_Time_Minutes'].sum()
                st.metric(
                    "⏱️ Toplam Eğitim Süresi", 
                    f"{total_time:.1f} dk", 
                    f"~{total_time/60:.1f} saat"
                )
    
    elif page == "🔍 Görüntü Sınıflandırma":
        st.header("🔍 Real-Time Görüntü Sınıflandırma")
        
        # Modelleri yükle
        models_dict, device = load_models()
        
        if not models_dict:
            st.error("❌ Hiçbir model yüklenemedi! Lütfen model dosyalarının varlığını kontrol edin.")
            return
        
        st.success(f"✅ {len(models_dict)} model başarıyla yüklendi!")
        
        # Görüntü yükleme
        uploaded_file = st.file_uploader(
            "Bir görüntü yükleyin (JPG, JPEG, PNG):",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Görüntüyü göster
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📷 Yüklenen Görüntü")
                st.image(image, caption="Orijinal Görüntü", use_column_width=True)
                
                # Görüntü bilgileri
                st.write(f"**Boyut:** {image.size}")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mod:** {image.mode}")
            
            with col2:
                st.subheader("🤖 Tahmin Sonuçları")
                
                # Tahmin yap
                with st.spinner("Modeller tahmin yapıyor..."):
                    predictions = predict_image(image, models_dict, device)
                
                # Sonuçları göster
                for model_name, pred in predictions.items():
                    with st.expander(f"📊 {model_name} Sonuçları", expanded=True):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                "🎯 Tahmin", 
                                pred['class_name_tr'].title(),
                                f"{pred['confidence']*100:.2f}%"
                            )
                        
                        with col_b:
                            st.metric(
                                "🔤 İngilizce", 
                                pred['class_name'].title(),
                                ""
                            )
                
                # Karşılaştırma grafiği
                st.subheader("📈 Detaylı Karşılaştırma")
                fig = create_prediction_chart(predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # En iyi tahmin
                best_pred = max(predictions.items(), key=lambda x: x[1]['confidence'])
                st.success(f"🏆 **En Güvenli Tahmin:** {best_pred[0]} - {best_pred[1]['class_name_tr'].title()} ({best_pred[1]['confidence']*100:.2f}%)")
    
    elif page == "📈 Model Karşılaştırması":
        st.header("📈 Model Performans Karşılaştırması")
        
        results_df = load_model_results()
        
        if results_df is None:
            st.error("❌ Model karşılaştırma sonuçları bulunamadı!")
            return
        
        # Performans tablosu
        st.subheader("📊 Detaylı Performans Tablosu")
        
        # Tabloyu güzelleştir
        styled_df = results_df.copy()
        styled_df['Best_Accuracy'] = styled_df['Best_Accuracy'].apply(lambda x: f"{x:.2f}%")
        styled_df['Final_Accuracy'] = styled_df['Final_Accuracy'].apply(lambda x: f"{x:.2f}%")
        styled_df['Training_Time_Minutes'] = styled_df['Training_Time_Minutes'].apply(lambda x: f"{x:.1f} dk")
        
        styled_df.columns = ['Model', 'En İyi Doğruluk', 'Final Doğruluk', 'Eğitim Süresi']
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Karşılaştırma grafikleri
        fig = create_model_comparison_chart(results_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Model analizi
        st.subheader("🔍 Model Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Performans Sıralaması:**")
            sorted_models = results_df.sort_values('Best_Accuracy', ascending=False)
            for i, (_, row) in enumerate(sorted_models.iterrows()):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
                st.write(f"{medal} **{row['Model']}**: {row['Best_Accuracy']:.2f}%")
        
        with col2:
            st.write("**⚡ Hız Sıralaması:**")
            sorted_speed = results_df.sort_values('Training_Time_Minutes', ascending=True)
            for i, (_, row) in enumerate(sorted_speed.iterrows()):
                medal = ["🚀", "⚡", "🏃"][i] if i < 3 else "🐌"
                st.write(f"{medal} **{row['Model']}**: {row['Training_Time_Minutes']:.1f} dk")
        
        # Öneriler
        st.subheader("💡 Öneriler")
        
        best_accuracy_model = results_df.loc[results_df['Best_Accuracy'].idxmax(), 'Model']
        fastest_model = results_df.loc[results_df['Training_Time_Minutes'].idxmin(), 'Model']
        
        st.info(f"""
        **🎯 En İyi Doğruluk için:** {best_accuracy_model} modelini kullanın.
        
        **⚡ Hızlı Sonuçlar için:** {fastest_model} modelini tercih edin.
        
        **⚖️ Denge için:** ResNet18 hem iyi performans hem de makul eğitim süresi sunar.
        """)

if __name__ == "__main__":
    main()