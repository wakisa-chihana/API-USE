from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes: bytes) -> np.array:
    # Open image from byte data
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert image to numpy array
    image_array = np.array(image)
    return image_array
