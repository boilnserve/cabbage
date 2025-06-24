from typing import Tuple, List, Any
from PIL import Image
import base64
from io import BytesIO

def resize_image(image, size: Tuple[int, int]) -> Image.Image:
    """Resize an image to the given size, maintaining aspect ratio. Args: image: PIL Image object. size: Target size as (width, height). Returns: Resized PIL Image object."""
    image = image.convert("RGB")
    image.thumbnail(size)
    return image

def extract_visuals(doc: dict) -> List[Any]:
    """Extract and resize images from a document for model input. Args: doc: Document dictionary containing 'images'. Returns: List of resized images."""
    visuals = [resize_image(image, (512, 512)) for image in doc.get('images',[])]
    if len(visuals) > 1:
        visuals = [resize_image(img, (256, 256)) for img in visuals]
    return visuals
    
def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string. Args: image: PIL Image object. Returns: Base64-encoded PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")