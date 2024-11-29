import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer

# Load the saved model
model = tf.keras.models.load_model('model/Offline_Handwritten.h5')

# Load the LabelBinarizer (make sure it is the same one used during training)
LB = LabelBinarizer()
LB.fit(["your", "label", "classes"])  # Fit the label binarizer with your labels during training

def predict_image(image_path):
    """
    Function to predict a handwritten character from an image.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (32, 32))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Predict the character class
    pred = model.predict(img)
    pred_label = LB.inverse_transform(pred)
    
    return pred_label[0]  # Return the predicted label

def extract_characters(image_path):
    """
    Extracts individual characters from the image using contour detection.
    """
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to fill gaps and make contours easier to detect
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    characters = []
    for c in cnts:
        if cv2.contourArea(c) > 10:  # Ignore small contours
            x, y, w, h = cv2.boundingRect(c)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)
            
            # Predict the character
            pred = model.predict(thresh)
            pred_label = LB.inverse_transform(pred)
            characters.append(pred_label[0])  # Add the predicted character to the list

    return characters
