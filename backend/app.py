from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import logging
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
CORS(app=app)

# Logging Configuration
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if not app.logger.handlers:
    app.logger.addHandler(handler)

logging.getLogger('src.datascience.pipeline.prediction_pipeline').setLevel(logging.INFO)

@app.route('/', methods=['GET']) # Route to display homepage
def homepage():
    return "<h1>Flask Backend is running!</h1><p>This is the backend for the react app</p>"

@app.route('/train', methods=['GET']) # Route to train the pipeline
def training():
    try:
        # Running training model pipeline
        app.logger.info("Starting training process....")
        os.system("python3 main.py")
        app.logger.info("Training Successful!")
        return jsonify({"message": "Training successful!"}), 200
    except Exception as e:
        app.logger.info("Error Occured when training....")
        return jsonify({"error": f"Training Failed :{str(e)}"}), 500

@app.route('/predict', methods=['POST']) # Route from web UI
def predict_data():
    """Route for receiving data from react and doing prediction"""
    app.logger.info("Starting Prediction prosess")

    if not request.is_json:
        app.logger.warning("The Request must be JSON")
        return jsonify({'error': 'request must be JSON'}), 400
    
    try:
        data_from_react = request.get_json(force=True, silent=False) # get JSON data from body request
        app.logger.info(f"Succesfully parsed JSON data : {data_from_react}")

    except Exception as e:
        app.logger.info(f"Failed to parse JSON Body...")
        return jsonify({'error': f'invalid JSON format in request body: {str(e)}'}), 400

    if not data_from_react:
        app.logger.warning("Empty JSON data received after successfully parsing.")
        return jsonify({'error': 'No JSON data received from request body.'}), 400

    try:
        app.logger.info(f"Starting Extraction from response...")
        Pregnancies = float(data_from_react['pregnancies'])
        Glucose = float(data_from_react['glucose'])
        BloodPressure = float(data_from_react['bloodPressure'])
        SkinThickness = float(data_from_react['skinThickness'])
        Insulin = float(data_from_react['insulin'])
        BMI = float(data_from_react['bmi'])
        DiabetesPedigree = float(data_from_react["diabetesPedigree"])
        Age = float(data_from_react['age'])
        
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigree, Age]).reshape(1, -1)
        
        obj = PredictionPipeline()
        prediction_result = obj.predict(input_data)
        
        app.logger.info(f"Prediction Successful. result: {prediction_result}")
        return jsonify({'prediction': prediction_result.tolist()}), 200
    
    except ValueError as ve:
        app.logger.error(f"Missing or Invalid Numerical data: {str(ve)}")
        return jsonify({'error': f'invalid data type: {str(ve)}, all input must be numeric'}), 400 
    
    except AttributeError as ae:
        app.logger.error(f"Invalid data type for fields :{str(ae)}")
        return jsonify({'error': f'missing one or more required input fields. please complete the data: {str(ae)}'}), 400
    
    except Exception as e:
        app.logger.critical(f"Unexpected error occured : {str(e)}")
        return jsonify({'error': f'An error occured when prediction: {str(e)}'}), 500

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)