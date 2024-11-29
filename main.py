from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import preprocess_image
from model import predict_image
import io

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    
    # Preprocess the image
    image = preprocess_image(image_bytes)
    
    # Get the prediction from the model
    prediction = predict_image(image)
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": prediction})

