# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:40:04 2024

@author: shamm
"""
import numpy as np
import pandas as pd

from flask import *
import pickle
import os
import re

app=Flask(__name__)
model=pickle.load(open('customers.pkl','rb'))
brand_name_le=pickle.load(open('brand_name_le.pkl','rb'))
food_category_le=pickle.load(open('food_category_le.pkl','rb'))
promotion_name_le=pickle.load(open('promotion_name_le.pkl','rb'))
store_city_le=pickle.load(open('store_city_le.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/service-details',methods=['POST','GET'])
def predictionpage():
    if request.method=='POST':
        store_sqft=request.form["store_sqft"]
        grocery_sqft=request.form["grocery_sqft"]
        brand_name=request.form["brand_name"]
        food_category=request.form["food_category"]
        promotion_name=request.form["promotion_name"]
        units_per_case=request.form["units_per_case"]
        net_weight=request.form["net_weight"]
        store_city=request.form["store_city"]
        alllist=[store_sqft,grocery_sqft,brand_name,food_category,promotion_name,units_per_case,net_weight,store_city]
        alllist[2] = brand_name_le.transform([alllist[2]])[0]
        #print(food_category_le.classes_)
        alllist[3] = food_category_le.transform([alllist[3]])[0]
        alllist[4] = promotion_name_le.transform([alllist[4]])[0]
        alllist[7] = store_city_le.transform([alllist[7]])[0]
        pred=[alllist]
        output=model.predict(pred)[0]
        print(output)
        return render_template('starter-page.html',output=output)
    return render_template('service-details.html')

if __name__=='__main__':
    app.run(host="127.0.0.9", port=5000, debug=True)