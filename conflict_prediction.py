# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import plotly.express as px

# 1. Veri Hazırlığı
# Geçmiş çatışma verilerini içeren bir veri seti kullanıyoruz
# Verisetinde tarih, bölge, ekonomik göstergeler, siyasi istikrar gibi çeşitli özellikler bulunmaktadır

data = pd.read_csv('conflict_data.csv')

# Eksik verileri temizleme
data = data.dropna()

# Özellik ve hedef değişkenlerin belirlenmesi
X = data.drop(['conflict'], axis=1)  # Özellikler: ekonomik göstergeler, siyasi istikrar, vb.
y = data['conflict']  # Hedef: çatışma var mı yok mu (1: var, 0: yok)

# 2. Veriyi Eğitim ve Test Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modelin Oluşturulması ve Eğitilmesi
# Basit bir Random Forest sınıflandırıcı kullanıyoruz
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Modelin Test Edilmesi
# Test verisi kullanılarak tahmin yapılması
y_pred = model.predict(X_test)

# Modelin başarımının değerlendirilmesi
accuracy = accuracy_score(y_test, y_pred)

# Streamlit arayüzü
st.title('Yapay Zeka Destekli Çatışma Erken Uyarı Sistemi')

# Modelin doğruluk oranını gösterme
st.write(f'Model Doğruluğu: {accuracy * 100:.2f}%')

# Kullanıcıdan yeni bir veri girişi alma
st.sidebar.header('Yeni Gözlem İçin Veri Girişi')
yil = st.sidebar.number_input('Yıl', min_value=2000, max_value=2100, value=2024)
bolge = st.sidebar.text_input('Bölge', 'Bölge_A')
ekonomik_gosterge = st.sidebar.number_input('Ekonomik Gösterge', min_value=0.0, max_value=10.0, value=3.2)
siyasi_istikrar = st.sidebar.number_input('Siyasi İstikrar Skoru', min_value=0.0, max_value=1.0, value=0.8)
diger_ozellik = st.sidebar.number_input('Diğer Özellik', min_value=0, max_value=10, value=1)

# Yeni veriyle tahmin yapma
new_data = pd.DataFrame([[yil, bolge, ekonomik_gosterge, siyasi_istikrar, diger_ozellik]], columns=X.columns)
conflict_risk = model.predict(new_data)

# Tahmin sonucunu gösterme
st.write('Çatışma Riski:', 'Var' if conflict_risk[0] == 1 else 'Yok')

# Veri görselleştirme
st.header('Veri Görselleştirme')
fig = px.scatter(data, x='ekonomik_gosterge', y='siyasi_istikrar', color='conflict', title='Ekonomik Gösterge ve Siyasi İstikrar İlişkisi')
st.plotly_chart(fig)
