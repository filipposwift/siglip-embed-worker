import runpod
from src.image_io import load_image_from_url
from src.model import encode_images, encode_texts

"""
ESEMPIO DI RICHIESTA AL SERVERLESS ENDPOINT - MODALITÀ IMMAGINI:

POST /runsync
Content-Type: application/json

{
  "input": {
    "mode": "images",
    "payload": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.png",
      "https://example.com/image3.jpg"
    ]
  }
}

RISPOSTA DI SUCCESSO IMMAGINI (status: 'OK'):

{
  "status": "OK",
  "embeddings": [
    {
      "image_url": "https://example.com/image1.jpg",
      "vector": [0.123, 0.456, 0.789, ...]  // Array di float normalizzati L2
    },
    {
      "image_url": "https://example.com/image2.png",
      "vector": [0.234, 0.567, 0.890, ...]
    }
  ]
}

ESEMPIO DI RICHIESTA - MODALITÀ TESTO:

POST /runsync
Content-Type: application/json

{
  "input": {
    "mode": "text",
    "payload": {
      "text": "bikini top with floral pattern",
      "id": "query_1"
    }
  }
}

RISPOSTA DI SUCCESSO TESTO (status: 'OK'):

{
  "status": "OK",
  "embeddings": [
    {
      "text_id": "query_1",
      "vector": [0.123, 0.456, 0.789, ...]  // Array di float normalizzati L2
    }
  ]
}

RISPOSTA CON ERRORI PARZIALI (status: 'PARTIAL'):

{
  "status": "PARTIAL",
  "embeddings": [
    {"image_url": "https://example.com/image1.jpg", "vector": [...]}
  ],
  "failed_urls": [
    {"url": "https://example.com/bad.jpg", "error": "404 Not Found"}
  ]
}

RISPOSTA CON ERRORE TOTALE (status: 'ERROR'):

{
  "status": "ERROR",
  "error": "Messaggio di errore descrittivo"
}
"""


def handler(event):
    """
    Handler principale per RunPod Serverless.
    
    Supporta due modalità:
    - mode="images": Riceve una lista di URL immagini, le scarica, le preprocessa
      e genera embedding usando SigLIP2.
    - mode="text": Riceve un oggetto con testo e id, genera embedding di testo
      usando SigLIP2 (stesso spazio vettoriale delle immagini per similarity search cross-modal).
    
    Args:
        event: Dizionario con chiave "input" contenente:
            - "mode": "images" o "text" (default "images" per retrocompatibilità)
            - "payload": 
              * Se mode="images": lista di URL immagini
              * Se mode="text": oggetto {"text": "...", "id": "..."}
        
    Returns:
        Dizionario con chiave "embeddings" contenente:
            - Per mode="images": lista di dict con "image_url" e "vector"
            - Per mode="text": lista di dict con "text_id" e "vector"
    """
    try:
        # Estrai l'input dall'evento
        input_data = event.get("input", {})
        
        # Determina la modalità (default "images" per retrocompatibilità)
        mode = input_data.get("mode", "images")
        
        # Supporto retrocompatibilità: se non c'è mode ma c'è "images", usa quello
        if mode not in ["images", "text"]:
            # Se mode non valido ma c'è "images" nel payload, assume mode="images"
            if "images" in input_data:
                mode = "images"
                payload = input_data.get("images", [])
            else:
                return {
                    "status": "ERROR",
                    "error": "Il campo 'mode' deve essere 'images' o 'text'"
                }
        else:
            # Leggi payload (o fallback a "images" per retrocompatibilità)
            payload = input_data.get("payload")
            if payload is None and mode == "images":
                # Retrocompatibilità: prova a leggere "images" se payload non esiste
                payload = input_data.get("images", [])
        
        if payload is None:
            return {
                "status": "ERROR",
                "error": "Il campo 'payload' è richiesto"
            }
        
        # Gestione modalità immagini
        if mode == "images":
            if not isinstance(payload, list):
                return {
                    "status": "ERROR",
                    "error": "Il campo 'payload' deve essere una lista di URL quando mode='images'"
                }
            
            if not payload:
                return {
                    "status": "ERROR",
                    "error": "Nessuna immagine fornita. Fornire una lista di URL in event['input']['payload']"
                }
            
            # Scarica e preprocessa le immagini
            images = []
            failed_urls = []
            
            for url in payload:
                try:
                    image = load_image_from_url(url)
                    images.append(image)
                except Exception as e:
                    failed_urls.append({"url": url, "error": str(e)})
            
            if not images:
                return {
                    "status": "ERROR",
                    "error": "Nessuna immagine valida elaborata",
                    "failed_urls": failed_urls
                }
            
            # Genera gli embedding in batch
            embeddings = encode_images(images)
            
            # Costruisci la risposta
            # Mappa solo gli URL che hanno generato embedding con successo
            successful_urls = [url for url in payload if url not in [f["url"] for f in failed_urls]]
            
            response = {
                "status": "PARTIAL" if failed_urls else "OK",
                "embeddings": [
                    {"image_url": url, "vector": embedding.tolist()}
                    for url, embedding in zip(successful_urls, embeddings)
                ]
            }
            
            # Aggiungi informazioni sugli URL falliti se presenti
            if failed_urls:
                response["failed_urls"] = failed_urls
            
            return response
        
        # Gestione modalità testo
        elif mode == "text":
            if not isinstance(payload, dict):
                return {
                    "status": "ERROR",
                    "error": "Il campo 'payload' deve essere un oggetto con 'text' e 'id' quando mode='text'"
                }
            
            if "text" not in payload or "id" not in payload:
                return {
                    "status": "ERROR",
                    "error": "Il campo 'payload' deve contenere 'text' (stringa) e 'id' (stringa) quando mode='text'"
                }
            
            text = payload.get("text")
            text_id = payload.get("id")
            
            if not isinstance(text, str) or not text.strip():
                return {
                    "status": "ERROR",
                    "error": "Il campo 'text' deve essere una stringa non vuota"
                }
            
            if not isinstance(text_id, str):
                return {
                    "status": "ERROR",
                    "error": "Il campo 'id' deve essere una stringa"
                }
            
            # Genera embedding per il testo
            embeddings = encode_texts([text])
            
            # Costruisci la risposta
            response = {
                "status": "OK",
                "embeddings": [
                    {"text_id": text_id, "vector": embeddings[0].tolist()}
                ]
            }
            
            return response
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": f"Errore interno durante l'elaborazione: {str(e)}"
        }


# Avvia il serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

