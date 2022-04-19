import json
import random
from collections import defaultdict


def split_train_dev(raw_fn, dev_size=200):
    # Random sample dev set of dev_size from raw LC-Quad train set
    with open(raw_fn) as fin:
        raw_dat_all = json.load(fin)
    all_ids = [entry['_id'] for entry in raw_dat_all]
    dev_set = random.sample(all_ids, dev_size)
    train_set = list(set(all_ids) - set(dev_set))
    assert(sorted(dev_set + train_set) == sorted(all_ids))
    return train_set, dev_set


def statistics_of_hop(raw_fn, split_fn):
    with open(split_fn) as fin_split, open(raw_fn) as fin_raw:
        split_dat = json.load(fin_split)
        train_set, dev_set = split_dat['train'], split_dat['dev']
        raw_dat_all = json.load(fin_raw)
    id2entry_raw = {entry['_id']:entry for entry in raw_dat_all}

    dev_cnt = defaultdict(lambda :0)
    for dev_id in dev_set:
        entry = id2entry_raw[dev_id]
        cur_len = len(entry['triples'])
        dev_cnt[cur_len] += 1

    train_cnt = defaultdict(lambda :0)
    for train_id in train_set:
        entry = id2entry_raw[train_id]
        cur_len = len(entry['triples'])
        train_cnt[cur_len] += 1

    for k in dev_cnt:
        dev_cnt[k] /= len(dev_set)
    for k in train_cnt:
        train_cnt[k] /= len(train_set)
    for k in sorted(train_cnt):
        print(k, train_cnt[k], dev_cnt[k])


if __name__ == '__main__':
    raw_fn = 'annotation/post_processed/Train_full_data_annotated.json'
    split_fn = 'data/meta/train_dev_split.json'
    dev_size = 200
    train_set, dev_set = split_train_dev(raw_fn, dev_size)
    #with open(split_fn, 'w') as fout:
    #    json.dump({'train': train_set, 'dev': dev_set}, fout)
    
    statistics_of_hop(raw_fn, split_fn)
