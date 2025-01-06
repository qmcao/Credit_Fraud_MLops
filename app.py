
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(

            income=float(request.form.get('income')),
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name'),
            age=int(request.form.get('customer_age')),
            bank_months_count=int(request.form.get('bank_months_count')),
            
            housing_status=request.form.get('housing_status'),
            current_address_months_count=int(request.form.get('current_address_months_count')),
            proposed_credit_limit=int(request.form.get('proposed_credit_limit')),
            employment_status=request.form.get('employment_status'),
            email_is_free=int(request.form.get('email_is_free')),
            has_other_cards = int(request.form.get('has_other_cards'))
            
            

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictionPipeline()
        print("Mid Prediction")
        results= predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
"""
test_backend.py

A standalone script to test the CustomData and PredictionPipeline classes
outside of the Flask application.
"""

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

def test_customdata_prediction():
    """
    Creates dummy input data for the CustomData class, 
    transforms it into a DataFrame, and runs the PredictionPipeline.
    """

    # 1. Create a mock CustomData object with sample values
    data = CustomData(
        # Numeric income in decile form [0.1 - 0.9]
        income=0.3,

        # Strings for first and last name
        first_name="John",
        last_name="Doe",

        # Age between 10 and 90
        age=35,

        # Bank months count between 0 and 32
        bank_months_count=12,

        # Housing status: one of ['BC', 'BE', 'BD', 'BA', 'BB', 'BF', 'BG']
        housing_status="BC",

        # Current address months count
        current_address_months_count=24,

        # Proposed credit limit between 200 and 2000
        proposed_credit_limit=1500,

        # Employment status: one of ['CB', 'CA', 'CC', 'CF', 'CD', 'CE', 'CG']
        employment_status="CA",

        # email_is_free: 0 (No) or 1 (Yes)
        email_is_free=1,

        # has_other_cards: 0 (No) or 1 (Yes)
        has_other_cards=0
    )

    # 2. Convert to DataFrame
    pred_df = data.get_data_as_data_frame()
    print("Constructed DataFrame:\n", pred_df, "\n")

    # 3. Instantiate the prediction pipeline
    predict_pipeline = PredictionPipeline()

    # 4. Make a prediction
    results = predict_pipeline.predict(pred_df)

    # 5. Print the prediction result
    print("Prediction result:", results)


    #test_customdata_prediction()

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
    #test_customdata_prediction()
    

