# SigLIP2 Image Embedding Worker - RunPod Serverless

Endpoint RunPod Serverless per generare embedding di immagini usando il modello SigLIP2, ottimizzato per similarity search su costumi da bagno con pattern e texture complessi.

## 🎯 Caratteristiche

- **Modello**: `google/siglip2-large-patch16-512` (risoluzione 512x512 per massimi dettagli)
- **Embedding normalizzati L2**: Pronti per cosine similarity search
- **Batch processing**: Elabora multiple immagini in parallelo
- **Gestione errori robusta**: Continua l'elaborazione anche se alcune immagini falliscono
- **Ottimizzato per GPU**: Utilizza CUDA su GPU NVIDIA (4090 o superiore)

## 📋 Requisiti

- RunPod Serverless account
- GPU NVIDIA con CUDA 12.8 (4090, A5000, RTX 50, ecc.)
- Repository GitHub collegato a RunPod

## 🏗️ Architettura

### Struttura del Progetto

```
siglip-embed-worker/
├── Dockerfile              # Immagine Docker per RunPod
├── requirements.txt        # Dipendenze Python
├── rp_handler.py          # Handler principale RunPod Serverless
├── src/
│   ├── __init__.py
│   ├── model.py           # Caricamento e inferenza SigLIP2
│   └── image_io.py        # Download e preprocessing immagini
└── README.md
```

### Scelte Architetturali

#### 1. **Modello SigLIP2 Large 512x512**

**Scelta**: `google/siglip2-large-patch16-512` invece di `siglip2-base-patch16-256`

**Motivazione**:
- **Dettagli superiori**: Risoluzione 512x512 cattura pattern e texture più fini
- **Similarity search migliore**: Embedding più ricchi e discriminanti per distinguere costumi simili
- **Caso d'uso specifico**: Per costumi da bagno con pattern complessi, i dettagli extra sono cruciali

**Trade-off**:
- ✅ Qualità embedding significativamente migliore
- ⚠️ Modello più pesante (~2-4x più lento, più memoria)
- ⚠️ Costo RunPod leggermente superiore

#### 2. **Resize Intermedio a 2048px**

**Scelta**: Resize intermedio a max 2048px prima del processor SigLIP2

**Motivazione**:
- **Protezione memoria**: Immagini enormi (4000x6000) vengono ridotte a ~8MB invece di 69MB
- **Dettagli massimi**: Per immagini tipiche (~2700px) perdita solo ~24% vs 62% con 1024px
- **Zero perdita per immagini piccole**: Immagini < 2048px non vengono toccate

**Flusso**:
```
Immagine originale (es. 1800x2699 o 4000x6000)
    ↓
Resize intermedio a max 2048px (mantiene dettagli massimi)
    ↓
AutoProcessor SigLIP2 resize a 512x512 (con padding/aspect ratio)
    ↓
Embedding normalizzato L2
```

**Analisi**:
- Immagine tipica 1800x2699: 13.9MB → 8MB (perdita 24%)
- Immagine grande 4000x6000: 69MB → 8MB (risparmio 88%)

#### 3. **Normalizzazione L2 degli Embedding**

**Scelta**: Normalizzare tutti gli embedding a norma unitaria (L2)

**Motivazione**:
- **Cosine similarity = dot product**: Con embedding normalizzati, `np.dot(emb1, emb2)` = cosine similarity
- **Performance database vettoriali**: Molti database (Pinecone, Weaviate, Qdrant) ottimizzano per vettori normalizzati
- **Stabilità numerica**: Riduce problemi di overflow/underflow
- **Best practice**: Standard per embedding (CLIP, SigLIP, ecc.)

**Implementazione**:
```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms  # Norma unitaria
```

#### 4. **Singleton Pattern per il Modello**

**Scelta**: Caricare modello e processor una sola volta (variabili globali)

**Motivazione**:
- **Performance**: Evita ricaricamento a ogni richiesta (modello ~1-2GB)
- **Memoria**: Un solo modello in memoria invece di multipli
- **Velocità**: Prima richiesta più lenta (download modello), successive molto veloci

**Implementazione**:
```python
processor = None
model = None

def _load_model():
    global processor, model
    if processor is None or model is None:
        # Carica solo se non già caricato
```

#### 5. **Device Sempre CUDA**

**Scelta**: `device = torch.device("cuda")` senza fallback CPU

**Motivazione**:
- **RunPod garantisce GPU**: I pod vengono sempre attivati con GPU NVIDIA (4090 o superiore)
- **Performance**: CUDA è 10-100x più veloce per inferenza deep learning
- **Semplicità**: Nessun codice condizionale necessario

#### 6. **Gestione Errori Robusta**

**Scelta**: Continuare l'elaborazione anche se alcune immagini falliscono

**Motivazione**:
- **Resilienza**: Un URL non valido non blocca l'intero batch
- **Informazioni utili**: Restituisce warning con dettagli degli URL falliti
- **UX migliore**: L'utente riceve embedding per immagini valide + info su quelle fallite

**Implementazione**:
```python
for url in image_urls:
    try:
        image = load_image_from_url(url)
        images.append(image)
    except Exception as e:
        failed_urls.append({"url": url, "error": str(e)})
```

## 🚀 Deployment su RunPod Serverless

### 1. Preparazione Repository GitHub

1. Push del codice su GitHub
2. Assicurarsi che la struttura sia corretta:
   ```
   ├── Dockerfile
   ├── requirements.txt
   ├── rp_handler.py
   └── src/
       ├── __init__.py
       ├── model.py
       └── image_io.py
   ```

### 2. Configurazione RunPod

1. Accedi a [RunPod Console](https://www.runpod.io/console/serverless)
2. Vai a **Serverless** → **New Endpoint**
3. Seleziona **GitHub Integration** e collega il repository
4. Configurazione:
   - **Handler**: `rp_handler.handler`
   - **Dockerfile path**: `Dockerfile` (root del repo)
   - **GPU**: NVIDIA 4090 o superiore (minimo 16GB VRAM)
   - **Container disk**: 20GB+ (per cache modello Hugging Face)

### 3. Build e Deploy

RunPod costruirà automaticamente l'immagine Docker dal Dockerfile. Il primo deploy può richiedere 5-10 minuti per:
- Build immagine Docker
- Download modello SigLIP2 da Hugging Face (~1-2GB)
- Setup ambiente

## 📡 Utilizzo API

### Endpoint

```
POST https://api.runpod.ai/v2/{endpoint_id}/runsync
```

### Richiesta

```json
{
  "input": {
    "images": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.png",
      "https://example.com/image3.jpg"
    ]
  }
}
```

### Risposta di Successo

```json
{
  "embeddings": [
    {
      "image_url": "https://example.com/image1.jpg",
      "vector": [0.123, 0.456, 0.789, ...]
    },
    {
      "image_url": "https://example.com/image2.png",
      "vector": [0.234, 0.567, 0.890, ...]
    }
  ]
}
```

### Risposta con Errori Parziali

```json
{
  "embeddings": [
    {
      "image_url": "https://example.com/image1.jpg",
      "vector": [0.123, 0.456, ...]
    }
  ],
  "warnings": {
    "failed_urls": [
      {
        "url": "https://example.com/bad.jpg",
        "error": "404 Not Found"
      }
    ],
    "message": "1 immagine/i non sono state elaborate"
  }
}
```

### Esempio con cURL

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "images": [
        "https://www.samelosangeles.com/cdn/shop/files/Same1260.jpg"
      ]
    }
  }'
```

### Esempio con Python

```python
import requests

endpoint_id = "your-endpoint-id"
api_key = "your-api-key"

response = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "input": {
            "images": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg"
            ]
        }
    }
)

result = response.json()
for embedding in result["embeddings"]:
    print(f"URL: {embedding['image_url']}")
    print(f"Vector dimension: {len(embedding['vector'])}")
```

## 🔍 Similarity Search

Gli embedding sono normalizzati L2, quindi puoi usare **dot product** per cosine similarity:

```python
import numpy as np

# Embedding normalizzati L2
emb1 = np.array(result["embeddings"][0]["vector"])
emb2 = np.array(result["embeddings"][1]["vector"])

# Cosine similarity = dot product (perché normalizzati)
similarity = np.dot(emb1, emb2)

# Oppure usa cosine_similarity di sklearn
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([emb1], [emb2])[0][0]
```

## 📊 Specifiche Tecniche

- **Dimensione embedding**: Dipende dal modello (tipicamente 768-1152 per SigLIP2 large)
- **Formato**: Array di float32 normalizzati L2
- **Batch size**: Supporta batch di qualsiasi dimensione (limitato solo da memoria GPU)
- **Timeout download**: 10 secondi per immagine
- **Risoluzione input**: 512x512 pixel (gestito automaticamente dal processor)

## 🐛 Troubleshooting

### Errore: "CUDA out of memory"

**Causa**: Batch troppo grande o modello troppo pesante per la GPU

**Soluzione**:
- Riduci il numero di immagini per batch
- Usa GPU con più VRAM (24GB+ invece di 16GB)

### Errore: "Model not found"

**Causa**: Modello non scaricato da Hugging Face

**Soluzione**:
- Verifica connessione internet nel container
- Controlla che `MODEL_NAME` in `src/model.py` sia corretto
- Il modello viene scaricato automaticamente al primo utilizzo

### Embedding inconsistenti

**Causa**: Possibile problema con normalizzazione L2

**Soluzione**:
- Verifica che gli embedding abbiano norma ~1.0
- Controlla che non ci siano vettori nulli

## 📝 Note

- Il modello viene scaricato automaticamente da Hugging Face al primo utilizzo
- La cache del modello viene salvata nel container (persiste tra richieste)
- Il primo utilizzo dopo deploy è più lento (download modello)
- Le richieste successive sono molto veloci (modello già in memoria)

## 🔗 Riferimenti

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless)
- [SigLIP2 Paper](https://arxiv.org/abs/2502.14786)
- [Hugging Face SigLIP2](https://huggingface.co/google/siglip2-large-patch16-512)

