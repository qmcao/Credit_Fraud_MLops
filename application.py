
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

if __name__=="__main__":
    app.run(host="0.0.0.0")
    #test_customdata_prediction()
    

