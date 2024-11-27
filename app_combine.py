from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Initialize the Flask app
app = Flask(__name__, template_folder='template')

# Set the model paths
LUNG_CANCER_MODEL_PATH = 'C:/Users/HP/Desktop/Lungs detection dataset/lung_cancer_model.h5'
LUNG_DISEASE_MODEL_PATH = 'lung_disease_detection_model.h5'

# load the modal you have save through cnn 
lung_cancer_model = load_model(LUNG_CANCER_MODEL_PATH)
lung_disease_model = load_model(LUNG_DISEASE_MODEL_PATH)

# Define categories
lung_disease_categories = ["Normal", "Corona Virus Disease", "Pneumonia", "Tuberculosis"]
lung_cancer_categories = ["Negative", "Positive"]

# This is the function to make prediction
def predict_lung_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = lung_disease_model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    return lung_disease_categories[pred_class]

# Function to make predictions for lung cancer
def predict_lung_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = lung_cancer_model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    return lung_cancer_categories[pred_class]

# Route for the home page (index2.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # Save the file to a static folder
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Choose the correct model based on user input
            model_type = request.form.get('model_type')
            if model_type == 'lung_cancer':
                prediction = predict_lung_cancer(file_path)
            else:
                prediction = predict_lung_disease(file_path)

            # Redirect to the result page with prediction and image path
            return redirect(url_for('result', prediction=prediction, image_path=file_path))
    
    return render_template('index2.html')

# Route for the result page
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    image_path = request.args.get('image_path')
    return render_template('result.html', prediction=prediction, image_path=image_path)

# Route for the summary page
@app.route('/summary')
def summary():
    prediction = request.args.get('prediction', 'No disease')  # Default to 'No disease' if no prediction is provided
    return render_template('summary.html', prediction=prediction)


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
