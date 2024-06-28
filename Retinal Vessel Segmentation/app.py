from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from generator import generator
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
UPLOAD_FOLDER = 'uploads/'
SEGMENTED_FOLDER = 'segmented/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the file (dummy processing step here)
        segmented_image_path = process_image(file_path, filename)

        return redirect(url_for('uploaded_file', filename=filename))


def process_image(file_path, filename):

    segmented_image_path = os.path.join(app.config['SEGMENTED_FOLDER'], filename)
    os.makedirs(app.config['SEGMENTED_FOLDER'], exist_ok=True)

    image_gen = generator(file_path)
    saved_model = load_model('model.h5')
    predict_mask = saved_model.predict(image_gen)

    # Assuming the output of the model is a mask of the same size as the input
    mask = predict_mask[0, :, :, 0]  # remove batch dimension and get the first channel if multi-channel
    mask = (mask * 255).astype(np.uint8)  # convert to uint8 image

    # Save the mask as an image
    mask_image = Image.fromarray(mask)
    mask_image.save(segmented_image_path)

# Route to display the uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['SEGMENTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
