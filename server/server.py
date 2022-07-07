import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

def get_model():
    global model
    model = load_model(r'E:/Automatic_Optical_Inspection_System/server/model/model.h5')
    print("Model loaded!")

def load_image(img_path):

    img = image.load_img(img_path)
    img_tensor = image.img_to_array(img) 
    img_tensor = cv2.GaussianBlur(img_tensor, (5,5), 0) 
    img_tensor = cv2.medianBlur(img_tensor, 5)
    _, img_tensor = cv2.threshold(img_tensor, 50, 255, cv2.THRESH_BINARY)
    img_tensor = cv2.resize(img_tensor, (224, 224))               
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.

    return img_tensor

def prediction(img_path):
    global predict_value
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0

    final=np.array(labels)

    predict_value = max(pred[0][0], pred[0][1])*100
    
    if final[0][0]==1:
        return "Bad"
    else:
        return "Good"

get_model()

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'E:/Automatic_Optical_Inspection_System/Server/images/', filename) 
        file.save(file_path)
        product = prediction(file_path)
        response = jsonify({
        'product': product,
        'predict_value': predict_value
        })
        print(product, predict_value)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == "__main__":
    app.run()