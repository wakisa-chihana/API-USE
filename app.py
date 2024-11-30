from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import shutil
import os
from model import extract_characters, get_word  # Import the necessary functions

# Initialize FastAPI
app = FastAPI()

# Directory to store uploaded images
UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure CORS to allow requests from all origins (you can restrict this if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific domains if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Custom 404 error handler
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"message": "The resource you are looking for was not found."}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

# You can also handle other types of validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Validation error occurred.", "errors": exc.errors()}
    )

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
