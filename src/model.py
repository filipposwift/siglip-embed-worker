import torch
import numpy as np
import os
from transformers import AutoProcessor, AutoModel
from typing import List, Optional
from PIL import Image

MODEL_NAME = "google/siglip2-large-patch16-512"

# Device sempre CUDA - il pod RunPod viene attivato con GPU NVIDIA (4090 o superiore)
device = torch.device("cuda")

# Path cache RunPod per modelli cached
RUNPOD_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

# Carica modello e processor una sola volta (variabili globali)
processor = None
model = None


def find_cached_model_path(model_name: str) -> Optional[str]:
    """
    Cerca il modello nella cache di RunPod.
    
    I modelli cached sono memorizzati in /runpod-volume/huggingface-cache/hub/
    seguendo la struttura: models--ORG--MODEL/snapshots/VERSION_HASH/
    
    Args:
        model_name: Nome del modello (es. "google/siglip2-large-patch16-512")
        
    Returns:
        Path completo al modello cached se trovato, None altrimenti
    """
    # Converti formato: "google/siglip2-large-patch16-512" -> "models--google--siglip2-large-patch16-512"
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(RUNPOD_CACHE_DIR, f"models--{cache_name}", "snapshots")
    
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            # Restituisci il path al primo snapshot (di solito c'è solo uno)
            model_path = os.path.join(snapshots_dir, snapshots[0])
            print(f"✅ Modello cached trovato in: {model_path}")
            return model_path
    
    return None


def _load_model():
    """
    Carica il modello e il processor se non sono già stati caricati.
    
    Cerca prima nella cache di RunPod (se configurato come cached model),
    altrimenti scarica da Hugging Face.
    """
    global processor, model
    if processor is None or model is None:
        # Cerca prima nella cache di RunPod
        cached_path = find_cached_model_path(MODEL_NAME)
        
        if cached_path:
            # Usa il modello cached (cold start molto più veloce)
            print(f"📦 Caricamento modello da cache RunPod: {cached_path}")
            processor = AutoProcessor.from_pretrained(cached_path)
            # Usa device_map="auto" come nell'esempio ufficiale Hugging Face
            model = AutoModel.from_pretrained(cached_path, device_map="auto").eval()
        else:
            # Fallback: scarica da Hugging Face (più lento al primo utilizzo)
            print(f"🌐 Modello non trovato in cache, download da Hugging Face: {MODEL_NAME}")
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            # Usa device_map="auto" come nell'esempio ufficiale Hugging Face
            model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto").eval()
        
        # model.eval() già chiamato in from_pretrained con device_map="auto"
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
    # Seguendo l'esempio ufficiale Hugging Face:
    # https://huggingface.co/google/siglip2-large-patch16-512
    inputs = proc(images=pil_images, return_tensors="pt", padding=True)
    
    # Sposta tutti gli input sul device del modello (come nell'esempio ufficiale)
    # device_map="auto" gestisce automaticamente il device, quindi usiamo model.device
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    
    # Forward pass senza calcolo gradienti
    # Usa get_image_features() passando **inputs (tutto il dict) come nell'esempio ufficiale
    # Questo è il metodo corretto per estrarre solo embedding immagini da SigLIP2
    with torch.no_grad():
        if hasattr(mdl, 'get_image_features'):
            # Metodo ufficiale: passa **inputs (tutto il dict) come nell'esempio Hugging Face
            image_embeds = mdl.get_image_features(**inputs)
        else:
            # Fallback: se get_image_features non è disponibile, prova altri metodi
            # Questo non dovrebbe essere necessario per SigLIP2
            raise ValueError("get_image_features() non disponibile. Verifica la versione di transformers.")
    
    # Converti in numpy array float32
    embeddings = image_embeds.cpu().numpy().astype(np.float32)
    
    # Normalizzazione L2 per similarity search (cosine similarity)
    # Normalizza ogni vettore lungo l'asse delle features (axis=1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Evita divisione per zero (vettori nulli)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms
    
    return embeddings

