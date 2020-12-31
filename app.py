from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import json
import warnings
import pickle
warnings.filterwarnings("ignore")
import h5py    
import numpy as np 

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])

def index():
    print('at 1')
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    print("At 2")
    data = request.form['search']
    # print("Data:::",data)
    model = load_model('model01.h5')
    # model = hf.get('train.py')
    with open('01_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
    seq = tokenizer.texts_to_sequences([data])   
    # print("seq::",seq)  

    Data = pad_sequences(sequences=seq, maxlen=20)
    # print("DATA:::",Data)
    prob = model.predict(Data)
    print("PROBABILITY::",prob)
    if prob[0][0] > 0.5:
        sentiment = "Positive"
        return render_template('predict.html',pred='Your Comment is {}'.format(sentiment))
    
    else:
        sentiment = "Negative"
        return render_template('predict.html',pred='Your Comment is {}'.format(sentiment))
    

    
    
if __name__ == '__main__':

    app.run(debug=True, port=3000)
