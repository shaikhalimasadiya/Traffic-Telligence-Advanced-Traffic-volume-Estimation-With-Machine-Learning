# Lines 1-9: Importing necessary libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 
import sklearn

# Line 12: Initialize the Flask application
app = Flask(__name__)

# Line 13: Load the trained ML model
model = pickle.load(open('model.pkl', 'rb'))  # Ensure model.pkl is in your project folder

# Line 16: Route to homepage (GET)
@app.route('/')
def home():
    # Line 18: Render the input form page
    return render_template('index.html')

# Line 16 again: Route to handle prediction (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Create input DataFrame
        input_data = {
            'temp': [temp],
            'rain': [rain],
            'snow': [snow],
            'weather': [weather],
            'holiday': [holiday],
            'year': [year],
            'month': [month],
            'day': [day],
            'hours': [hours],
            'minutes': [minutes],
            'seconds': [seconds]
        }

        input_df = pd.DataFrame(input_data)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('result.html', prediction_text=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")



# Line 36: Run the app
if __name__ == '__main__':
    app.run(debug=True)
