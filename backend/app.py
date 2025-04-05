from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('data/leaf_model.h5')

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy'
]

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return "🌿 Leaf Disease Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    try:
        processed_image = preprocess_image(filepath)
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
