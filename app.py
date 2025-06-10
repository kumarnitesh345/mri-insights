import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import imutils
# Remove requests as APIs are no longer used
# import requests
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model('brain_tumor_model.h5')

# Load hospital data from JSON file
try:
    with open('hospitals_top_30_plus_cities_india.json', 'r') as f:
        hospital_data = json.load(f)
except FileNotFoundError:
    hospital_data = {}
    print("Error: hospitals_top_30_plus_cities_india.json not found. Hospital suggestions will not be available.")
except json.JSONDecodeError:
    hospital_data = {}
    print("Error: Could not decode JSON from hospitals_top_30_plus_cities_india.json. Hospital suggestions will not be available.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/detect')
def detect():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            img = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(img)
            # Handle different output shapes
            if prediction.shape[-1] == 1:
                confidence = float(prediction[0][0])  # For sigmoid output
            else:
                confidence = float(prediction[0][1])  # For softmax output (2 classes)
            
            result = {
                'prediction': 'Tumor Detected' if confidence > 0.5 else 'No Tumor Detected',
                'confidence': f'{confidence * 100:.2f}%',
                'image_path': filepath
            }
            
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type'})
    except Exception as e:
        import traceback
        print('Exception in /predict:', traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# New route to get supported cities for suggestions
@app.route('/cities', methods=['GET'])
def get_cities():
    return jsonify(list(hospital_data.keys()))

# New route to suggest hospitals from JSON data
@app.route('/suggest_hospitals', methods=['POST'])
def suggest_hospitals():
    data = request.get_json()
    location = data.get('location')

    if not location:
        return jsonify({'error': 'Location not provided'}), 400

    # Find the city in the hospital data (case-insensitive)
    matched_city_key = None
    for city_key in hospital_data.keys():
        # Simple check if the input location is in the city key (case-insensitive)
        if location.lower().strip() in city_key.lower():
             matched_city_key = city_key
             break # Found a potential match, take the first one

    # Return the list of hospital objects if city is found, otherwise an empty list
    suggestions = []
    if matched_city_key and matched_city_key in hospital_data:
        suggestions = hospital_data[matched_city_key]
    # No need for an 'else' with a dummy message here, the frontend will handle the empty list case.

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True) 