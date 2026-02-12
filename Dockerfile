# Mantemos a base NVIDIA para suporte a GPU
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Configurações para evitar prompts interativos e arquivos temporários
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalação do Python 3.12 e dependências de sistema
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ajusta os links simbólicos para que o sistema use o 3.12 por padrão
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

# Instala o pip especificamente para o Python 3.12 (forma mais robusta)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Copia e instala as dependências do seu projeto
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "src/gui/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.fileWatcherType=none"]
    
# FROM python:3.12-slim

# # Evita arquivos .pyc e logs presos
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt ./
# COPY . .

# RUN pip3 install -r requirements.txt

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# # ENTRYPOINT ["streamlit", "run", "src/gui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CMD ["streamlit", "run", "src/gui/app.py", \
#     "--server.port=8501", \
#     "--server.address=0.0.0.0", \
#     "--server.enableCORS=false", \
#     "--server.enableXsrfProtection=false", \
#     "--server.fileWatcherType=none"]