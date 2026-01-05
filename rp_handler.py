import runpod
from src.image_io import load_image_from_url
from src.model import encode_images

"""
ESEMPIO DI RICHIESTA AL SERVERLESS ENDPOINT:

POST /runsync
Content-Type: application/json

{
  "input": {
    "images": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.png",
      "https://example.com/image3.jpg"
    ]
  }
}

RISPOSTA DI SUCCESSO (status: 'OK'):

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
  "error": "Nessuna immagine fornita. Fornire una lista di URL in event['input']['images']"
}
"""


def handler(event):
    """
    Handler principale per RunPod Serverless.
    
    Riceve un evento con una lista di URL immagini, le scarica, le preprocessa
    e genera embedding usando SigLIP2.
    
    Args:
        event: Dizionario con chiave "input" contenente "images" (lista di URL)
        
    Returns:
        Dizionario con chiave "embeddings" contenente lista di dict con "image_url" e "vector"
    """
    try:
        # Estrai l'input dall'evento
        input_data = event.get("input", {})
        image_urls = input_data.get("images", [])
        
        # Validazione input
        if not image_urls:
            return {
                "status": "ERROR",
                "error": "Nessuna immagine fornita. Fornire una lista di URL in event['input']['images']"
            }
        
        if not isinstance(image_urls, list):
            return {
                "status": "ERROR",
                "error": "Il campo 'images' deve essere una lista di URL"
            }
        
        # Scarica e preprocessa le immagini
        images = []
        failed_urls = []
        
        for url in image_urls:
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
        successful_urls = [url for url in image_urls if url not in [f["url"] for f in failed_urls]]
        
        response = {
            "status": "PARTIAL" if failed_urls else "OK",
            "embeddings": [
                {"image_url": url, "vector": embedding.tolist()}
                for url, embedding in zip(successful_urls, embeddings)
            ]
        }
        
        # Aggiungi informazioni sugli URL falliti se presenti (senza message verboso)
        if failed_urls:
            response["failed_urls"] = failed_urls
        
        return response
        
    except Exception as e:
        return {
            "status": "ERROR",
            "error": f"Errore interno durante l'elaborazione: {str(e)}"
        }


# Avvia il serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

