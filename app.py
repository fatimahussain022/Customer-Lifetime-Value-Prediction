#Import libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pandas as pd

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
    model = tf.keras.models.load_model('./Trained Model/my_model.h5')
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
    array = preprocess_data(input_generator)
    model = get_model()
    prediction = model.predict([array])
    print(prediction)

    return render_template('index.html', datetime=datetime, prediction_placeholder=0)
 
def preprocess_data(inputs):
    '''
    Takes the input generator of html elements,
    Converts the data into 3 groups,
    pass the dataframe back to predict fucntion 
    '''
    # convert generator to list
    inputs = list(inputs)

    # couple the corresponding values, first being monthly count
    # second being monthly expense
    list_of_tuples = []
    for i in range(0, len(inputs), 2):
        list_of_tuples.append((int(inputs[i]),int(inputs[i+1])))
    print(list_of_tuples)
    input_len = len(list_of_tuples)

    # convert the data into groups of 3
    NUM_GROUPS = 3
    per_group = len(list_of_tuples)/NUM_GROUPS 
    print(per_group) 
    df = pd.DataFrame(data=list_of_tuples, columns=['count','expense'])
    grouped_df = df.groupby(df.index // per_group).apply(lambda x: pd.Series({'count': x['count'].sum(), 'expense': x['expense'].sum()}))
    grouped_df['average'] = grouped_df['expense']/grouped_df['count']
    array = np.reshape(np.array(grouped_df), (9,))

    return array



if __name__ == "__main__":
    app.run(debug=True) #debug=True means you won't have to run the server again & again, it'll update directly for you
