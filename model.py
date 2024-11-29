import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
 
# Load the pre-trained model (ensure it's saved in the correct path)
model = model = tf.keras.models.load_model('handwritten_model.h5')


def predict_image(image: np.array):
    # Reshape the image to match model input (28x28x1)
    image = image.reshape(-1, 28, 28, 1)  # Reshape to 28x28 image with 1 channel
    image = image.astype('float32') / 255  # Normalize the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability
    return int(predicted_class[0])  # Return the predicted digit
