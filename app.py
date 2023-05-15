#Import libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import string

from flask import Flask, request, render_template
from datetime import datetime
from dateutil.relativedelta import relativedelta

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html', datetime=datetime)
    # return render_template('index.html', current_month=current_month, current_year=current_year, datetime=datetime, relativedelta=relativedelta)



def get_model():
    '''
    Loading model for the server-side
    '''
    model = tf.keras.models.load_model('./templates/my_model.h5')
    print(" * Model loaded!")
    return model


@app.route('/predict', methods=['POST'])
def predict():
    '''
    A view for rendering results on HTML GUI
    '''

    print("\n\nIn PREDICT \n\n")
    # get the html inputs
    input_generator = request.form.values()

    # preprocess the html inputs
    preprocessed_input = preprocess_data(input_generator)


    return render_template('index.html', datetime=datetime, prediction_placeholder=0)
 
def preprocess_data(input_generator):
    '''
    Takes the input generator of html elements,
    Converts the data into 3 groups,
    pass the dataframe back to predict fucntion 
    '''
    input = list(input_generator)
    
    # convert the data into groups of 3
    per_group = len(input)/3

    return input_generator



if __name__ == "__main__":
    app.run(debug=True) #debug=True means you won't have to run the server again & again, it'll update directly for you
