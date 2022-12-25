from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model=joblib.load('iris_model_LR.pkl')
scaler=joblib.load('scaler.save')

app =Flask(__name__)

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def home():
    if request.method =='POST':
        sl=request.form['SepalLength']
        sw = request.form['SepalWidth']
        pl = request.form['PetalLength']
        pw = request.form['PetalWidth']
        data = np.array([[sl, sw, pl, pw]])
        x = scaler.transform(data)
        print(x)
        prediction = model.predict(x)
        print(prediction)
        image=prediction[0]+'.png'
        image=os.path.join(app.config['UPLOAD_FOLDER'],image)
    return render_template('index.html',prediction=prediction[0],image=image)


if __name__ == '__main__':
    app.run(debug=True)