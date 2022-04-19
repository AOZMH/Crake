import json
import sys
sys.path.append('../annotation')
from preprocess import parse_sparql


# Preliminaries of ASK sentences
ask_pre_set = ('did', 'do', 'does', 'is', 'was', 'were')
# Keywords of COUNT sentences
count_key_set_1 = ("count ", "Count ")
count_key_set_2 = ("how many", "how much", "total number", "number of ")


def judge_ask(ques):
    # Judge if is a ASK query, otherwise a SELECT query
    pre_ques = ques.strip().split(' ')[0].lower()
    return pre_ques in ask_pre_set


def judge_count(ques):
    # Judge if is a COUNT query, otherwise no aggregation function is used
    ques = ques.strip()
    for count_key in count_key_set_1:
        if count_key in "$" + ques:
            return True
    lower_ques = ques.strip().lower()
    for count_key in count_key_set_2:
        if count_key in lower_ques:
            return True
    
    return False


def test_ask():
    dir_name = '../annotation/post_processed/'
    full_train_fn = dir_name + 'Train_full_data_annotated.json'
    with open(full_train_fn) as fin:
        train_full = json.load(fin)
    
    for dat in train_full:
        spq = dat['raw_sparql']
        ques = dat['question']
        is_ask = judge_ask(ques)
        res = parse_sparql(spq)
        if res['type'] != 'ask':
            assert(res['type'] == 'select')
            if is_ask:
                print(ques, spq)
        else:
            if not is_ask:
                print(ques, spq)


def test_count():
    dir_name = '../annotation/post_processed/'
    full_train_fn = dir_name + 'Train_full_data_annotated.json'
    with open(full_train_fn) as fin:
        train_full = json.load(fin)
    
    for dat in train_full:
        spq = dat['raw_sparql']
        ques = dat['question']
        is_count = judge_count(ques)
        res = parse_sparql(spq)
        if res['type'] == 'ask':
            continue
        assert(res['type'] == 'select')
        if res['function'] == 'COUNT':
            if not is_count:
                print('Not recalled')
                print(ques, spq, '\n')
        else:
            assert(res['function'] is None)
            if is_count:
                print('Not precise')
                print(ques, spq, '\n')


if __name__ == '__main__':
    test_ask()
    test_count()
