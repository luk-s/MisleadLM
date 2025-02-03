import sys
import io
import contextlib
from io import StringIO
from tqdm import tqdm
from pebble import ProcessPool
from multiprocess.pool import Pool
import multiprocess
from concurrent.futures import TimeoutError
from copy import deepcopy
import ast
import signal
from tqdm import tqdm

def handler(signum, frame):
   raise Exception("Timeout")

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

@contextlib.contextmanager
def stdinIO(input_str):
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(input_str)
    yield
    sys.stdin = old_stdin
    
def preprocess_input(input_data):
    if isinstance(input_data, list):
        return '\n'.join(input_data)
    else:
        return input_data
    
def worker(code, input_data=''):
    input_data = preprocess_input(input_data)
    with stdinIO(input_data), stdoutIO() as s:
        try:
            exec(code, {'__name__':"__main__"})
            res = s.getvalue().strip()
        except SystemExit as e:
            res = s.getvalue().strip() 
    return res

def execute_code(code):
    preds = []
    with ProcessPool(1) as p:
        async_results = [p.schedule(worker, (code,), timeout=2)]
        for result in async_results:
            pred = '[error] default pred'
            try:
                pred = result.result()
            except TimeoutError:
                pred = '[error] Execution time of the program exceeds 2 second, terminate execution.'
            except Exception as e:
                error = e
                pred = f'[error] {error}'
            preds.append(pred)
    pred = preds[0]
    return pred

def is_float(text):
    try:
        text = float(text)
        return True
    except:
        return False

def check(input, pred, answer, special_judge=None):   
    try:
        pred = pred.strip()
        answer = answer.strip()
    except:
        return False
    
    if pred == answer:
        return True
    
    if answer.startswith("[") and answer.endswith("]") and len(answer) < 1e6:
        try:
            answer = ast.literal_eval(answer)
            answer = '\n'.join(answer).strip()
        except:
            answer = answer
    
    if pred.startswith("[") and pred.endswith("]") and len(pred) < 1e6:
        try:
            pred = ast.literal_eval(pred)
            pred = '\n'.join(pred).strip()
        except:
            pass


    if special_judge is not None:
        flag = special_judge(input, pred, answer)
    else:
        try:
            pred = [i.strip() for i in pred.strip().split('\n')]
            answer = [i.strip() for i in answer.strip().split('\n')]
            flag = True
            for i_pred, i_answer in zip(pred, answer):
                if i_pred != i_answer:
                    if not (is_float(i_pred) and is_float(i_answer) and abs(float(i_pred) - float(i_answer)) < 1e-4):
                        flag = False
                        break
            return flag
        except:
            pass
        
        try:
            pred = float(pred)
            answer = float(answer)
            flag1 = abs(pred - answer) < (max(abs(answer), abs(pred)) * 1e-4)
            flag2 = abs(pred - answer) < 1e-4
            flag = flag1 | flag2
            return flag
        except:
            pass
        
        flag = pred == answer
    return flag    

def execute_code_test_case(question, code, cases):
    cases = deepcopy(cases)
    special_judge = None

    use_parallel = True

    
    with ProcessPool(20) as p:
        async_results = [p.schedule(worker, (code, test_case['input']), timeout=1) for test_case in cases]
        preds = []
        for result in async_results:
            pred = '[error] default pred'
            try:
                pred = result.result()
            except multiprocess.TimeoutError:
                pred = '[error] Execution time of the program exceeds 2 second, terminate execution.'
            except Exception as e:
                error = e
                print('error: ', e)
                pred = f'[error] {error}'
            preds.append(pred)
       
    for pred, test_case in zip(preds, cases):
        input_data = test_case['input']
        test_case['output'] = str(test_case['output']).strip()
        test_case['pred'] = pred
        test_case['flag'] = check(input_data, pred, test_case['output'], special_judge)
    
    overall_flag = all([i['flag'] for i in cases])
    return cases, overall_flag

def batched_execute_code_test_case(batched_data):
    special_judge = None

    with ProcessPool(16) as p:
        async_results = []
        for task_id, task in enumerate(batched_data):
            question, code, cases = task
            if 'thread' in code:
                code = ""
            for case_id, case in enumerate(cases):
                async_results.append(p.schedule(worker, (code, case['input']), timeout=1))

        preds = []
        for result in tqdm(async_results):
            pred = '[error] default pred'
            try:
                pred = result.result()
            except multiprocess.TimeoutError:
                pred = '[error] Execution time of the program exceeds 2 second, terminate execution.'
            except Exception as e:
                error = e
                pred = f'[error] {error}'
            preds.append(pred)

    batched_res = []
    pred_idx = 0
    for task_id, task in enumerate(batched_data):
        question, code, cases = task
        cases = deepcopy(cases)
        for case_id, test_case in enumerate(cases):
            input_data = test_case['input']
            test_case['output'] = str(test_case['output']).strip()
            test_case['pred'] = preds[pred_idx]
            test_case['flag'] = check(input_data, preds[pred_idx], test_case['output'], special_judge)
            pred_idx += 1
        overall_flag = all([i['flag'] for i in cases])
        batched_res.append((cases, overall_flag))

    return batched_res
