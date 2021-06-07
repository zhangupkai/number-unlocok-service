import numpy as np
from flask import Flask, request, jsonify
import joblib
import logging
import sys
import os

app = Flask(__name__)


@app.route('/')
def home():
    logging.info('Hello')
    print('HELLO', file=sys.stderr)
    return 'Hello World'


@app.route('/collect_and_judge', methods=['POST'])
def collect():
    print('collect')
    data = request.get_json()
    print(data)
    # 写入data.txt，追加模式'a'
    file = open('data/txt/data.txt', 'a')
    file.writelines(['\n', str(data['duration'])])
    file.close()

    # 写入collect.txt
    file = open('data/txt/collect.txt', 'a')
    file.writelines(['\n', str(data['duration']), ',',
                     str(data['sizeAtDown']), ',',
                     str(data['sizeAtUp']), ',',
                     str(data['sizeAvg']), ',',
                     str(data['pressureAtDown']), ',',
                     str(data['pressureAtUp']), ',',
                     str(data['pressureAvg'])])
    file.close()

    # 设置密码时反馈当前按键是重按还是轻按
    model = joblib.load('model/cluster.pkl')
    prediction = model.predict(np.mat([[data['duration'], data['sizeAvg']]]))
    output = {
        'result': int(prediction[0])
    }
    return jsonify(output)


@app.route('/scenes_collect_and_judge', methods=['POST'])
def scenes_collect_and_judge():
    print('collect')
    data = request.get_json()
    print(data)
    # 写入data.txt，追加模式'a'
    file = open('data/txt/scenes_collect.txt', 'a')
    file.writelines('\n')
    for duration in data['durationList']:
        file.writelines([str(duration), ','])
    file.close()

    output = {
        'result': str('0000')
    }
    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    print('predict')
    data = request.get_json()
    print('data', data)

    model = joblib.load('model/cluster.pkl')
    prediction = model.predict(np.mat([[data['duration'], data['sizeAvg']]]))
    output = {
        'result': int(prediction[0])
    }
    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
