import numpy as np
from flask import Flask, request, jsonify
import joblib
import logging
import sys

app = Flask(__name__)
model = joblib.load('cluster.pkl')


@app.route('/')
def home():
    logging.info('Hello')
    print('HHHHH', file=sys.stderr)
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
    print('Start!')
    data = request.get_json()
    print('data', data)
    duration = 0.0
    for v in data.values():
        duration = v
    prediction = model.predict(np.mat([[duration, 1]]))
    output = {
        'result': int(prediction[0])
    }
    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
