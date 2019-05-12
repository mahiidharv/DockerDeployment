# Import libraries
#import numpy as np
from app import app
import os
import pandas as pd
from flask import Flask, request, Response,render_template,flash
from flask_paginate import Pagination,get_page_args
from werkzeug import secure_filename
#import json
#app = Flask(__name__)
from joblib import load 
import configparser

# get the current directory location
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

config = configparser.ConfigParser()
config.read(APP_ROOT+'/'+'config.ini')

host=config['MySQL']['host']
user=config['MySQL']['user']
password=config['MySQL']['password']
db=config['MySQL']['db']
charset=config['MySQL']['charset']
port=config['MySQL']['port']

import mysql.connector
from sqlalchemy import create_engine
#connector = 'mysql+mysqlconnector://'+str(user)+':'+str(password)+'@'+str(host)+':'+str(port)+'/'+str(db)
connector = 'mysql+mysqlconnector://root:password@172.19.0.2:3306/insofe_customerdata'
print(connector)
engine = create_engine(connector, echo=False)
#engine = create_engine('mysql+mysqlconnector://root:password@127.0.0.1:3333/insofe_customerdata', echo=False)

#
# Load the model
model = load(APP_ROOT+'/pickle/model.pkl_2019-04-10 10:34:05.282838')


def get_pages(data,per_page=10,offset=0):
    return data[offset:offset+per_page]


def result(APP_ROOT,model,engine=engine):
    data = pd.read_csv(APP_ROOT+"/data/test_cases.csv")
    try:
        data = data.drop('y',axis=1)
        data = data.drop('Unnamed: 0',axis=1)
    except:
        pass
    data = data.reset_index(drop=True)
        
    prediction = model.predict(data)
    output = prediction
    output = output.tolist()
      
    data['prediction'] = output
    cols = list(data.columns)
    col=[cols[0]]+[cols[len(cols)-1]]+cols[1:len(cols)-1]
    data = data[col]
    outputcsv = data
    outputcsv.to_sql(name='predictions', con=engine, if_exists = 'replace', index=False)
    return outputcsv

@app.route('/')
def index():
   return render_template('upload.html')


@app.route('/upload',methods=['POST'])
def upload():
    
    target = os.path.join(APP_ROOT,'data/')
    #print(target)
    if not os.path.isdir(target):
      os.mkdir(target)
    
    file= request.files['file']

    if file:
      filename = secure_filename(file.filename)
      destination = '/'.join([target,filename])
      data = pd.read_csv(file)
      
      data.to_csv(destination)
      
       
            
    return render_template('layout.html')

		
@app.route('/submit',methods=['GET','POST'])
def submit():
    

    if request.form.getlist("download")==['on']:

        print(type(request.form.getlist("download")))
        
        outputcsv = result(APP_ROOT,model)
        
        #outputcsv.to_csv(destination)
        csv = outputcsv.to_csv()
        return Response(
                csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=result.csv"})
        
    else:

        print(type(request.form.getlist("display")))
        
        outputcsv = result(APP_ROOT,model)
        
        dat = outputcsv.to_dict(orient="records")
        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
        total = len(dat)
       
        pagination_pages = get_pages(data = dat,offset=offset, per_page=per_page)
        
        pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
        return render_template('submit.html',data = pagination_pages,page = page,per_page = per_page,pagination=pagination)

            
    


'''

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    test_data = request.json
    #print(test_data)
    #print(type(test_data))

    temp = pd.DataFrame.from_dict(test_data,orient='index')
    #print(type(temp))
    
    prediction = model.predict(temp.T)
    
    output = prediction
    output = output.tolist()[0]

    #return jsonify(output)
    return render_template("main/layout.html",output= output)


'''
