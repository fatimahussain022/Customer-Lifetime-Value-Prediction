# Customer-Lifetime-Value-Prediction
The objective of this project is to create a deep learning model that forecasts the Customer Lifetime Value (CLTV) of a retail company's customer base. By precisely predicting CLTV, companies can recognize customers who are high-value, customize marketing campaigns to them, and enhance their customer retention strategies. This will assist retail companies in determining whether it is more lucrative to retain existing customers or acquire new ones.


## Process Overview
1.   Study, Analyze, and Select the optimal features for Customer Lifetime Value Prediction.
2.   Define an appropriate time frame for Customer Lifetime Value calculation
3.   Calculate lifetime value for training the machine learning model
4.   Build and run the machine learning model
5.   Check if the model is useful


## Dataset
The dataset contains information on 54,000 transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online store. 
Attribute Information:
*   InvoiceNo: nominal - 6-digit unique number for each transaction. 
*   Stockcode: nominal - 5-digit unique number assigned to each product.
*   Description: nominal - Product name.
*   Quantity: numeric - the quantityof each product per transaction.
*   InvoiceDate: numeric - date and time of when each transaction was generated.
*   UnitPrice: numeric - the product price per unit. 
*   CustomerID: nominal - 5-digit integral number uniquely assigned to each customer.
*   Country: nominal - name of country where each customer resides.

Download Dataset from: https://www.kaggle.com/code/shailaja4247/customer-lifetime-value-prediction/data


Reference: https://medium.com/swlh/predict-customer-lifetime-value-with-machine-learning-545624073d14


## Jupyter Notebook Overview
The notebook is divided into two modules:

 &ensp; **Module 1: Preprocessing, Visualization and Model Selection**
 
The module is designed to take raw data as input and output a preprocessed dataset, along with visualizations and select a machine learning model that can be used for prediction.

 &ensp; **Module 2: Optimization**

This module is responsible for determining the best combination of hyperparameters to obtain the optimal machine learning model. This includes testing the best architecture for the Multilayer Neural Network, experimenting with different optimizers, and using grid search to determine the best hyperparameters.


## Using the Trained Model
To recreate the model, including its weights and the optimizer:
```
model = tf.keras.models.load_model('my_model.h5')
```

To make predictions:
```
predictions = model.predict(input_data)
```
where the input data should be of the shape: __(n, 9)__
given n is any number of samples.


## Executing the project
1. Ensure that you have anaconda installed on your device.
2. Create a virtual environment for this project "conda create –n cltv python=3.9"
3. Activate the environment "conda activate cltv"
4. Install the requiremnts "pip install -r requirement.txt"
5. Run the app.py file 
