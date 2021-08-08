import flask
from flask import request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def Pridiction_algo(age):
    data = pd.read_csv('Book.csv')
    X = data.drop(columns=['Genere'])
    Y = data['Genere']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    predict = classifier.predict([[age]])
    return predict

app = flask.Flask(__name__)
app.config["DEBUG"] = True
@app.route('/', methods=['GET'])
def home():
    return "<h1>Normal Route</h1>"
@app.route('/<int:age>', methods=['GET'])
def Api(age):
    # age = request.args.get('age')
    x = Pridiction_algo(age)
    return x[0]
#app.run()
# app.run(host="0.0.0.0", port=int("5555"), debug=True)
if __name__ == '__main__':
     app.run(port='5002',debug=True)
