from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf


from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
application = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="This is a diseased cotton leaf.Use Prophylactic sprays with copper or organic fungicides 2-3 times at 15 days interval."
    elif preds==1:
        preds="This is a diseased cotton plant.Use Seed treatment with organo -mercurials; removal of collateral weed hosts ; spraying plants 2 times with copper fungicides before boll formation."
    elif preds==2:
        preds="This is a fresh cotton leaf"
    elif preds==3:
        preds="This is a fresh cotton plant"
    
    return preds


@application.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    application.run(port=5001,debug=True)
