from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import numpy as np
from io import BytesIO
from model import extract_characters, get_word  # Assuming these functions are defined

app = FastAPI()

UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and get word prediction.
    This also returns the image with a bounding box around the detected text.
    """
    try:
        # Save the uploaded file
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract characters and predict the word (assuming this function works)
        characters = extract_characters(image_path)
        word = get_word(characters)  # Use the get_word function to join the characters
        
        # Detect the bounding box (this could be an enhancement in the extract_characters function)
        # For now, this is just a mock bounding box (you would replace this with actual detection logic)
        bbox = [(50, 50, 200, 50)]  # Example: [x, y, width, height]
        
        # Draw bounding box on the image
        modified_image_path = draw_bounding_box(image_path, bbox)

        # Open the modified image to send back
        with open(modified_image_path, "rb") as img_file:
            img_bytes = img_file.read()
        
        # Send the modified image back along with the predicted word
        response_data = {
            "predicted_word": word,
            "image": img_bytes
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

