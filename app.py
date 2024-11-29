from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from model import extract_characters, get_word  # Import the necessary functions

# Initialize FastAPI
app = FastAPI()

# Directory to store uploaded images
UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and get word prediction.
    """
    # Save the uploaded file
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract characters and predict the word
    characters = extract_characters(image_path)
    word = get_word(characters)  # Use the get_word function to join the characters
    
    return JSONResponse(content={"predicted_word": word})

# Optionally, a health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
