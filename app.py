#Import libraries
from flask import Flask, request, render_template
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# route
@app.route('/')
def home():
    return render_template('index.html', datetime=datetime, prediction_placeholder=0)


def get_model():
    '''
    Loading model for the server-side
    '''
    model = load_model('./Trained Model/881.h5')
    return model


@app.route('/predict', methods=['POST'])
def predict():
    '''
    A view for rendering results on HTML GUI with the prediction
    '''

    print("\n\nIn PREDICT \n\n")
    # get the html inputs
    input_generator = request.form.values()

    # preprocess the html inputs
    array = preprocess_data(input_generator)

    # get model and predict
    model = get_model()
    prediction = model.predict(array)

    # rednder with the predictions
    return render_template('index.html', datetime=datetime, prediction_placeholder=prediction[0][0])
 
def preprocess_data(inputs):
    '''
    Takes the input generator of html elements,
    Converts the data into 3 groups,
    pass the dataframe back to predict fucntion 
    '''
    # convert generator to list
    inputs = list(inputs)
    print(inputs)

    # couple the corresponding values, 
    # first being monthly count
    # second being monthly expense
    list_of_tuples = []
    for i in range(0, len(inputs), 2):
        list_of_tuples.append((int(inputs[i]),int(inputs[i+1])))

    # convert the data into groups of 3
    NUM_GROUPS = 3
    per_group = len(list_of_tuples)/NUM_GROUPS 

    # convert to dataframe, make 3 groups, get sum, count, average
    df = pd.DataFrame(data=list_of_tuples, columns=['count','expense'])
    grouped_df = df.groupby(df.index // per_group).apply(lambda x: pd.Series({'count': x['count'].sum(), 'expense': x['expense'].sum()}))
    grouped_df['average'] = grouped_df['expense']/grouped_df['count']
    grouped_df = grouped_df.fillna(0)
    print(grouped_df)

    # flatten
    array = np.reshape(np.array(grouped_df), (1, 9,))
    print(array)

    return array



if __name__ == "__main__":
    app.run(debug=True) #debug=True means you won't have to run the server again & again, it'll update directly for you
