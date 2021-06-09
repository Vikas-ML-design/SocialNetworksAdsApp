import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model(open('saved_model.pb', 'rb'))
scFeatures = pickle.load(open('CustomerCategoryFeatureMod.ft','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = float(request.form['age'])
    sal = float(request.form['sal'])
    stdFeatures = scFeatures.transform(age)
    predLabel = model.predict_classes(stdFeatures)      

    return render_template('index.html', prediction_text='Given customer is a {} customer".format("Good" if predLabel[0][0] == 1 else "Bad"))


if __name__ == "__main__":
    app.run(debug=True)
