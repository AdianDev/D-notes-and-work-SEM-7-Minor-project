from flask import Flask, redirect , url_for, render_template, request
import pickle
import sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer as cv
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
import string

app = Flask(__name__)

model=None
vec=None
with open("prsa_pickle", "rb") as f: 
    model=pickle.load(f)
    
with open("vect_pickle", "rb") as m: 
    vec=pickle.load(m)

goodRev="Product Review is Good.....❤❤"
badrev="Product Review is Bad.....!☹☹"
def pre(txt):
    # vec=cv()
    stra=[txt]
    rev=vec.transform(stra)
    res=model.predict(rev)[0]
    if(res==1):
        return goodRev
    else:
        return badrev
    
print(pre("this is worst product"))
    
@app.route("/",methods=['GET'])
def home():
    return render_template("index.html" )

@app.route("/sentiment",methods=['POST'])
def predict():
    if request.method == 'POST':
        txt=request.form['review']
        txt=pre(txt)
        if txt=="":
            return render_template('index.html',prediction_texts="please provide review first")
        else:
            return render_template('index.html',prediction_text=txt)
    else:
        return render_template('index.html')

    
if __name__ == "__main__":
    app.run(port=3000,debug=True)


