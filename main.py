from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)


# Load the prediction model architecture and weights
model_architecture_path = "handwriting_recognition_prediction_model.json"
model_weights_path = "handwriting_recognition_prediction_model.weights.h5"


# Load the model
with open(model_architecture_path, "r") as json_file:
    prediction_model_json = json_file.read()
prediction_model = tf.keras.models.model_from_json(prediction_model_json)
prediction_model.load_weights(model_weights_path)


# Charset used in the model
charset = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "


# Preprocessing function
def preprocess_image(image):
    height, width = image.shape
    processed_image = np.full((64, 256), 255, dtype=np.uint8)

    if width > 256:
        image = image[:, :256]

    if height > 64:
        image = image[:64, :]

    processed_image[:height, :width] = image
    return cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)


# Decode the labels function
def decode_labels(encoded):
    return ''.join([charset[ch] for ch in encoded if ch != -1])


# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Read the image in grayscale
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({'error': 'Invalid image file'})

    # Preprocess the image
    img = preprocess_image(img) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    # Predict using the model
    predictions = prediction_model.predict(img)

    # Get input length for CTC decoding
    input_length = np.ones(predictions.shape[0]) * predictions.shape[1]

    # Decode the predictions using CTC
    decoded_predictions = tf.keras.backend.get_value(
        tf.keras.backend.ctc_decode(predictions, input_length=input_length, greedy=True)[0][0])
    for i in range(0,len(decoded_predictions)):
        print(decode_labels(decoded_predictions[i]))
    # Decode labels
    decoded_label = decode_labels(decoded_predictions[0])

    return jsonify({'prediction': decoded_label})


# Route for the home page
@app.route('/')
def home():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True, port=5999)
