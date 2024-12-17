# Python 3.12 imajını baz alıyoruz
FROM python:3.11-slim

# Çalışma dizini oluşturun
WORKDIR /app

# Sistem bağımlılıklarını yüklemek için
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-distutils \  
    && rm -rf /var/lib/apt/lists/*

# Proje dosyalarını konteynere kopyala
COPY . /app

# Bağımlılıkları yükle
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit'i başlatmak için komut
CMD ["streamlit", "run", "app.py"]
