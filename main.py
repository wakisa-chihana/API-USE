from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model at startup
model = None

# Create FastAPI app
app = FastAPI()

# Function to load the model
def load_trained_model():
    global model
    model = tf.keras.models.load_model('model.h5')  # Ensure correct path

# Load model when the app starts
@app.on_event("startup")
async def startup_event():
    load_trained_model()

# Image preprocessing function
def preprocess_image(img: Image.Image):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize to range [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array

# Prediction function
def predict(img_array):
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Route to accept image input and return prediction
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes))

        # Preprocess the image
        img_array = preprocess_image(img)

        # Get prediction
        predicted_class = predict(img_array)

        # Return the prediction as JSON (convert numpy.int64 to Python int)
        return JSONResponse(content={"predicted_class": int(predicted_class)})

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
