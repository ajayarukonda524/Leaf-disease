from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('data/leaf_model.h5')

# Load class names from saved class_indices.json
with open('data/class_indices.json') as f:
    class_indices = json.load(f)

# Reverse the class_indices to get a list of class names in correct index order
class_names = [None] * len(class_indices)
for class_name, idx in class_indices.items():
    class_names[idx] = class_name

# Image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
