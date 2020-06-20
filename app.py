import os
import sys
import random
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import numpy as np
from util import base64_to_pil

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.applications.vgg16 import VGG16

# model = VGG16(include_top=True)

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model = model(classes=15)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)


    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)


    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

       
        preds = model_predict(img, model)

        # preds = decode_predictions(preds)
        print(preds)

       
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
     
       

        # print(pred_class) 
        print(pred_proba)


        return jsonify(result=pred_proba, probability=pred_proba)

    return None


if __name__ == '__main__':
   
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
