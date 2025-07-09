import os

import numpy as np

import cv2

import joblib

from flask import Flask, request, jsonify, send_from_directory

from flask_cors import CORS

from tensorflow.keras.models import load_model


app = Flask(__name__)

CORS(app)


# Load model components

cnn_model = load_model('cnn_model.h5') # Adjust path as needed

scaler = joblib.load('scaler1.pkl')

kmeans = joblib.load('kmeans_model1.pkl')


def load_image(image_path, size=(128, 128)):

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:

return None

img_resized = cv2.resize(img, size)

return img_resized.flatten().reshape(1, -1) # Flatten to (1, 16384)


def extract_features_cnn(image):

image = image.reshape((1, 128, 128, 1)) # Reshape for CNN input

features = cnn_model.predict(image)

return features


@app.route('/predict', methods=['POST'])

def predict():

if 'file' not in request.files:

return jsonify({'error': 'No file part'}), 400


file = request.files['file']

if file.filename == '':

return jsonify({'error': 'No selected file'}), 400


# Save the file to a temporary location

temp_path = 'temp_image.jpg'

file.save(temp_path)


try:

# Load and preprocess image

image = load_image(temp_path)

if image is None:

return jsonify({'error': 'Invalid image'}), 400


# Extract features and scale

new_image_features = extract_features_cnn(image)

new_image_scaled = scaler.transform(new_image_features)


# Predict cluster

predicted_cluster = kmeans.predict(new_image_scaled)


# Map cluster to label

label = 'CANCER' if predicted_cluster[0] == 0 else 'NON CANCER'

return jsonify({'prediction': label})

finally:

os.remove(temp_path) # Clean up the temporary file


@app.route('/')

def serve_index():

return send_from_directory('.', 'OCCAD.html') # Adjust if your HTML file is in a different location


if __name__ == '__main__':
app.run(debug=True)