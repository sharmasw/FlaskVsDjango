from flask import Flask
from flask import request, jsonify
app = Flask(__name__)
from sklearn.externals import joblib
import pandas as pd

loadedModel=joblib.load('randomforestModel.pkl')

@app.route('/scoreJson', methods=['POST'])
def scoreData():
    if request.method == 'POST':
        print (request.form)
        print (dict(request.form))
        dataToscore=pd.DataFrame(data=[dict(request.form)])
        score=loadedModel.predict(dataToscore)[0]
    return jsonify({'result':int(score)})



if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000,debug=True)