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
    print('HELLO', file=sys.stderr)
    return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
    print('Start!')
    data = request.get_json()
    print('data', data)
    # duration = 0.0
    # for v in data.values():
    #     duration = v

    # 写入data.txt，追加模式'a'
    file = open('data.txt', 'a')
    file.writelines(['\n', str(data['duration'])])
    file.close()

    # 写入collect.txt
    file = open('collect.txt', 'a')
    file.writelines(['\n', str(data['duration']), ',',
                     str(data['sizeAtDown']), ',',
                     str(data['sizeAtUp']), ',',
                     str(data['sizeAvg']), ',',
                     str(data['pressureAtDown']), ',',
                     str(data['pressureAtUp']), ',',
                     str(data['pressureAvg'])])
    file.close()

    prediction = model.predict(np.mat([[data['duration'], 1]]))
    output = {
        'result': int(prediction[0])
    }
    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
