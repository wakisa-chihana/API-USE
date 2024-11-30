from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import numpy as np
from io import BytesIO
from model import extract_characters, get_word  # Assuming these functions are defined

# Initialize FastAPI
app = FastAPI()

# Directory to store uploaded images
UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure CORS to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Root endpoint to display a welcoming message
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Handwritten Classification API!"}

# Function to draw a bounding box around the detected text
def draw_bounding_box(image_path, bbox):
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw the bounding box on the image
    for box in bbox:
        x, y, w, h = box  # box format: [x, y, width, height]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
    
    # Save the modified image
    modified_image_path = image_path.replace(".jpg", "_with_bbox.jpg")
    cv2.imwrite(modified_image_path, image)
    
    return modified_image_path

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and get word prediction.
    This also returns the image with a bounding box around the detected text.
    """
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

# Optionally, a health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
