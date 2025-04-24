from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = keras.models.load_model('my_model.h5')
scaler = joblib.load('scaler.save')

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and order inputs according to training features
    features = [
        float(data['Length']),
        float(data['Diameter']),
        float(data['Height']),
        float(data['Weight']),
        float(data['Sex_F']),
        float(data['Sex_I']),
        float(data['Sex_M'])
    ]

    # Scale input and predict
    input_array = np.array([features])  # shape: (1, 7)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0][0]

    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
