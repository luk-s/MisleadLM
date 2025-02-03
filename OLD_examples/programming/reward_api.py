import json
import argparse
from typing import List
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json

app = Flask("__name__")
CORS(app, origins="*")


from utils.executor import execute_code_test_case, batched_execute_code_test_case

def get_scores(sample: dict):
    eval_res = execute_code_test_case(None, sample['response'], sample['test_cases'])
    flags = [i['flag'] for i in eval_res[0]]
    return flags

def batched_get_scores(batched_data):
    batched_data = [(sample['query'], sample['response'], sample['test_cases']) for sample in batched_data]
    res_list = batched_execute_code_test_case(batched_data)
    flags = [[i['flag'] for i in eval_res[0]] for eval_res in res_list]
    return flags

 
@app.route('/unittest', methods=['POST'])
def get_reward():
    data = json.loads(request.data)
    flags = get_scores(data)
    return make_response(jsonify(flags))


@app.route('/batched_unittest', methods=['POST'])
def batched_get_reward():
    data = json.loads(request.data)
    flags = batched_get_scores(data)
    return make_response(jsonify({"flags": flags}))

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8118)
