from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('titanic_prediction_model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/result',methods=['POST'])

def home():
    pclass=request.form['pclass']
    sex=request.form['sex']
    age=request.form['age']
    sibsp=request.form['sibsp']
    parch=request.form['parch']
    fare=request.form['fare']
    embarked=request.form['embarked']
    inp=np.array([[pclass,sex,age,sibsp,parch,fare,embarked]])
    inp=inp.astype('int')
    prediction=model.predict(inp)
    died=np.array([0])
    survived=np.array([1])
    return render_template('result.html',data=prediction,value0=died,value1=survived)

if __name__=="__main__":
    app.run(debug=True)