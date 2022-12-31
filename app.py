from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)       

model=pickle.load(open('model.pkl','rb'))    #Opening the file in reading mode


@app.route('/')
def hello_world():
    return render_template("forest.html")     #It will render the initial page to web 


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]  #Getting the data that user has entered
    final=[np.array(int_features)]                        #converting the user entered data into an array
    print(int_features)
    print(final)                                          #Checking the data in console
    prediction=model.predict_proba(final)                 #getting the probability by sending the user enterd data  by passing it into the model
    output='{0:.{1}f}'.format(prediction[0][1], 2)        #formatting data for 2 decimal places

    if output>str(0.5):
        return render_template('forest.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('forest.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
