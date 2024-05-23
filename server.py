from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import pickle

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max request size to 16MB

def suppress_tf_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

suppress_tf_logs()

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Load the model and tokenizer
model_path = '/home/site/wwwroot/GRU_sentiment_model.h5'
tokenizer_path = '/home/site/wwwroot/tokenizer_turkish.pickle'
model = None
tokenizer = None


try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500

    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        processed_text = preprocess_text(text)

        print(text)

        sequences = tokenizer.texts_to_sequences([processed_text])
        padded_sequences = pad_sequences(sequences, maxlen=100)

        # Make prediction
        predictions = model.predict(padded_sequences)
        prediction = float(predictions[0][0])

        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # For Azure App Service
