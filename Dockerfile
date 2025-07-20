FROM python:3.11-slim

WORKDIR /app

# Zainstaluj potrzebne pakiety systemowe
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Instalacja narzędzi Pythonowych
RUN pip install --upgrade pip setuptools wheel build

# Skopiuj tylko plik zależności w pierwszym etapie – dla lepszego cache
COPY RiNALMo /app/RiNALMo
COPY . .

# Instalacja lokalnych zależności
RUN pip install . --no-deps
RUN pip install ./RiNALMo

# Główne zależności PyTorch i inne
RUN pip install torch==2.3.0 \
    torch-geometric==2.5.3 \
    torch-scatter==2.1.2+pt23cu121 \
    torch-sparse==0.6.18+pt23cu121 \
    torch-cluster==1.6.3+pt23cu121 \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    numpy==1.26.4 \
    scikit-learn>=1.4.0 \
    pandas \
    biopython>=1.83 \
    rnapolis==0.3.11 \
    einops \
    wandb \
    fastapi \
    uvicorn \
    python-multipart

# Otwórz port FastAPI
EXPOSE 8080

# Domyślnie uruchamiaj serwer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
