from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
from model import extract_characters, get_word, detect_bounding_boxes  # Import necessary functions for prediction

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

def draw_bounding_box(image_path, bboxes):
    """
    Draw bounding boxes on the image at the given path and save the modified image.
    
    Args:
    - image_path (str): Path to the uploaded image.
    - bboxes (list of tuples): List of bounding box coordinates (x, y, width, height).
    
    Returns:
    - str: Path to the saved image with bounding boxes drawn.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw bounding boxes (assuming bboxes is a list of (x, y, w, h))
    for (x, y, w, h) in bboxes:
        # Draw a rectangle: (x, y) is the top-left corner, (x+w, y+h) is the bottom-right corner
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, 2px thickness
    
    # Save the image with bounding boxes
    output_path = image_path.replace(".jpg", "_modified.jpg")  # You can change the extension if needed
    cv2.imwrite(output_path, image)
    
    return output_path

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

        # Extract characters and predict the word
        characters = extract_characters(image_path)
        word = get_word(characters)  # Use the get_word function to join the characters
        
        # Detect bounding boxes for the detected characters/words (Replace with actual detection)
        bboxes = detect_bounding_boxes(image_path)  # This should return a list of bounding boxes
        
        # Draw bounding box on the image
        modified_image_path = draw_bounding_box(image_path, bboxes)

        # Return the modified image file with a predicted word
        return JSONResponse(content={
            "predicted_word": word,
            "image_url": f"/images/{os.path.basename(modified_image_path)}"
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Route to serve the modified image
@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(UPLOAD_DIR, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

# Optionally, a health check endpoint
@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
