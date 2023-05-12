from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#loading the model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index_1.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    country = float(request.form.get('Country',0))
    ISO = float(request.form.get('ISO',0))
    sex = float(request.form.get('Sex',0))
    year = float(request.form.get('Year',0))
    ASDP = float(request.form.get('ASDP',0))
    lower_uncertainty = float(request.form.get('Lower_95_uncertainty',0))
    result = model.predict([[country, ISO, sex, year, ASDP, lower_uncertainty]])[0]
    return render_template('index.html', **locals())

@app.route('/EDA')
def EDA():
    return render_template('index_3.html')

@app.route('/PDA')
def PDA():
    return render_template('index_4.html')

if __name__ == '__main__':
    app.run(debug=True)
