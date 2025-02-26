from flask import Flask,render_template,request,jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
'''
it create an instance of Flask class , which will be your WSGI application.
'''
application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        temprature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform([[temprature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True,port=5001)