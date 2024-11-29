from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from model import model, LB, get_letters, get_word  # Import necessary functions

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
    try:
        # Save the uploaded file
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ensure the image is readable
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
        
        # Extract characters and predict the word
        letters, processed_image = get_letters(image_path)
        if not letters:
            raise HTTPException(status_code=400, detail="No characters detected or prediction failed.")
        
        word = get_word(letters)  # Join the characters to form the word
        
        # Optionally, delete the uploaded image after processing
        os.remove(image_path)
        
        return JSONResponse(content={"predicted_word": word})
    
    except Exception as e:
        # Handle errors and provide feedback
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/health/")
async def health_check():
    return {"status": "healthy"}