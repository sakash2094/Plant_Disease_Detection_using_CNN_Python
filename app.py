import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the model from the correct path
model_path = r'D:\Plant\plant_disease_detection-main\model.keras'  # Update with correct path
model = load_model(model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

# Define your labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Function to get predictions
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Route for homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Specify the uploads folder (ensure the folder exists in the current directory)
        uploads_folder = r'D:\Plant\plant_disease_detection-main\uploads'
        os.makedirs(uploads_folder, exist_ok=True)

        # Save the file to the uploads folder
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)

        # Get predictions
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
