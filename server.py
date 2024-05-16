from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import pickle

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max request size to 16MB

# Function to suppress TensorFlow logs
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

# Load the model
model_path = 'sentiment_analysis_model.h5'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the pre-fitted tokenizer
with open('tokenizer_turkish.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        processed_text = preprocess_text(text)

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
    app.run(host='0.0.0.0', port=5000)
