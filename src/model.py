import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from typing import List
from PIL import Image

MODEL_NAME = "google/siglip2-large-patch16-512"

# Device sempre CUDA - il pod RunPod viene attivato con GPU NVIDIA (4090 o superiore)
device = torch.device("cuda")

# Carica modello e processor una sola volta (variabili globali)
# Il modello viene scaricato automaticamente da Hugging Face al primo utilizzo
processor = None
model = None


def _load_model():
    """Carica il modello e il processor se non sono già stati caricati."""
    global processor, model
    if processor is None or model is None:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
    return processor, model


def encode_images(pil_images: List[Image.Image]) -> np.ndarray:
    """
    Genera embedding per una lista di immagini usando SigLIP2.
    
    Utilizza il modello large a 512x512 per catturare maggiori dettagli, ideale per
    similarity search su costumi da bagno con pattern e texture complessi.
    
    Gli embedding vengono normalizzati L2 per ottimizzare la similarity search.
    Con embedding normalizzati, il dot product è equivalente alla cosine similarity.
    
    Args:
        pil_images: Lista di oggetti PIL.Image
        
    Returns:
        np.ndarray di shape (N, D) dove N è il numero di immagini e D è la dimensione dell'embedding.
        Gli embedding sono normalizzati L2 (norma unitaria).
    """
    # Carica modello se necessario
    proc, mdl = _load_model()
    
    # Preprocessa le immagini in batch
    inputs = proc(images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass senza calcolo gradienti
    with torch.no_grad():
        outputs = mdl(**inputs)
        # SigLIP2 restituisce image_embeds direttamente
        image_embeds = outputs.image_embeds
    
    # Converti in numpy array float32
    embeddings = image_embeds.cpu().numpy().astype(np.float32)
    
    # Normalizzazione L2 per similarity search (cosine similarity)
    # Normalizza ogni vettore lungo l'asse delle features (axis=1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Evita divisione per zero (vettori nulli)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    
    return embeddings

