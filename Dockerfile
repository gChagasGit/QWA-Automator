# Usa Python 3.10 leve
FROM python:3.12-slim

# Evita arquivos .pyc e logs presos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema para o OpenCV funcionar no Linux (dentro do container)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define pasta de trabalho
WORKDIR /app

# Copia e instala as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto
COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando padrão: Inicia o Streamlit
CMD ["streamlit", "run", "src/gui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]