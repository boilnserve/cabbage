from typing import Tuple, List, Any
from PIL import Image
import base64
from io import BytesIO

def resize_image(image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size)
    return image

def extract_visuals(doc: dict) -> List[Any]:
        visuals = [resize_image(image, (512, 512)) for image in doc.get('images',[])]
        if len(visuals) > 1:
            visuals = [resize_image(img, (256, 256)) for img in visuals]
        return visuals
    
def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")