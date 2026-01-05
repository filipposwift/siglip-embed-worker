import requests
from PIL import Image
from io import BytesIO


def load_image_from_url(url: str) -> Image.Image:
    """
    Scarica un'immagine da un URL e la restituisce come oggetto PIL.Image.
    
    Args:
        url: URL dell'immagine da scaricare
        
    Returns:
        PIL.Image in formato RGB
        
    Raises:
        requests.RequestException: Se il download fallisce
        IOError: Se l'immagine non può essere aperta
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Resize intermedio: riduce il lato lungo a max 2048px mantenendo aspect ratio
    # Questo aiuta a ridurre l'uso di memoria per immagini molto grandi (>2048px).
    # Il processor SigLIP2 farà comunque il resize finale a 512x512, ma questo step
    # mantiene dettagli massimi (perdita solo ~24% per immagini tipiche ~2700px)
    # proteggendo da immagini enormi (4000x6000 → 8MB invece di 69MB).
    max_size = 2048
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image

