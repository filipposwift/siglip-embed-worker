# Base image RunPod con PyTorch 2.8 + CUDA 12.8
# NOTA: Se questa immagine non esiste, verificare su Docker Hub o documentazione RunPod ufficiale
# per il nome corretto dell'immagine base PyTorch 2.8 + CUDA 12.8
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Imposta la directory di lavoro
WORKDIR /app

# Copia e installa le dipendenze in un singolo layer per ottimizzare la cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY rp_handler.py .
COPY src/ ./src/

# Comando di avvio
CMD ["python", "-u", "rp_handler.py"]

