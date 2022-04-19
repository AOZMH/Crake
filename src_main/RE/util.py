# 利用preprocess_data预处理好的数据, 生成RE模型dataloader

import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, RobertaTokenizer


# Tokenizer Info
PAD_TOKEN_ID = None


def load_preprocessed_token_ids_full(train_full_fn, dev_full_fn, test_full_fn, sample_ratio):
    # 将 preprocess_data.py 中预处理好的数据(question, mention, relation的token id)载入
    # sample_ratio: 对tag=1的数据做up-sampling, 只对train做!
    
    with open(train_full_fn, 'rb') as f_train, open(dev_full_fn, 'rb') as f_dev, open(test_full_fn, 'rb') as f_test:
        train_toks, train_type_masks, train_tags = pickle.load(f_train)
        [dev_toks, dev_type_masks, dev_tags], dev_triple_ranges, raw_dev_set = pickle.load(f_dev)
        [test_toks, test_type_masks, test_tags], test_triple_ranges, raw_test_set = pickle.load(f_test)
    print("RE data loaded")
    
    # up-sampling
    appendix_tok, appendix_mask, appendix_tag = [], [], []
    for cur_tok, cur_mask, cur_tag in zip(train_toks, train_type_masks, train_tags):
        if cur_tag == 0:
            continue
        appendix_tok.extend([cur_tok]*sample_ratio)
        appendix_mask.extend([cur_mask]*sample_ratio)
        appendix_tag.extend([cur_tag]*sample_ratio)
    train_toks.extend(appendix_tok)
    train_type_masks.extend(appendix_mask)
    train_tags.extend(appendix_tag)
    print("Up-sampling completed")
    return [train_toks, train_type_masks, train_tags],\
            ([dev_toks, dev_type_masks, dev_tags], dev_triple_ranges, raw_dev_set),\
            ([test_toks, test_type_masks, test_tags], test_triple_ranges, raw_test_set)


class REDataset(Dataset):
    # RE sentence-pair-CLS 数据类型
    def __init__(self, all_data):
        assert(len(all_data[0]) == len(all_data[1]) == len(all_data[2]))
        self.ques = all_data[0]     # 由ques-mention1-mention2-relation组成
        self.type_masks = all_data[1]
        self.tags = all_data[2]
    
    def __len__(self):
        return len(self.ques)
    
    def __getitem__(self, idx):
        return self.ques[idx], self.type_masks[idx], self.tags[idx]


def RE_collate(batch):
    # batch = [dataset[ix1], dataset[ix2], ...] = [[tokens_i, type_masks_i, tags_i], ...]
    lst = []
    padded_seq = []
    type_mask = []
    att_mask = []
    re_tags = []

    max_seq_len = len(max(batch, key=lambda x:len(x[0]))[0])
    for seq, mask, tag in batch:
        assert(len(seq)==len(mask))
        padded_seq.append(seq + [PAD_TOKEN_ID]*(max_seq_len-len(seq)))
        type_mask.append(mask + [1]*(max_seq_len-len(mask)))
        att_mask.append([1]*len(seq) + [0]*(max_seq_len-len(seq)))
        re_tags.append(tag)

    lst.append(torch.LongTensor(padded_seq))
    lst.append(torch.LongTensor(type_mask))
    lst.append(torch.FloatTensor(att_mask))
    lst.append(torch.LongTensor(re_tags))

    return lst


def form_dataset(train_full_fn, dev_full_fn, test_full_fn, train_batch, test_batch, tokenizer, num_workers, sample_ratio):
    # 生成RE的train和test dataloader
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = tokenizer.pad_token_id
    train_all_data,\
        (dev_all_data, dev_triple_ranges, raw_dev_set),\
        (test_all_data, test_triple_ranges, raw_test_set) =\
        load_preprocessed_token_ids_full(train_full_fn, dev_full_fn, test_full_fn, sample_ratio)
    print('Tokens converted to ids.\nTrain / Test length: {} / {}.'.format(len(train_all_data[0]), len(test_all_data[0])))
    train_dataset, dev_dataset, test_dataset = REDataset(train_all_data), REDataset(dev_all_data), REDataset(test_all_data)
    
    # dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, collate_fn=RE_collate, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=test_batch, collate_fn=RE_collate, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, collate_fn=RE_collate, shuffle=False, num_workers=num_workers)
    print('Dataloaders formed.')
    return train_loader, dev_loader, test_loader, dev_triple_ranges, raw_dev_set, test_triple_ranges, raw_test_set


if __name__ == '__main__':
    train_full_fn = '../data/RE_gold/Train_RE_gold_preprocessed_1hop_split.json'
    dev_full_fn = '../data/RE_gold/Dev_RE_gold_preprocessed_1hop_split.json'
    test_full_fn = '../data/RE_gold/Test_RE_gold_preprocessed_1hop.json'

    train_batch, test_batch = 16, 4
    num_workers = 0
    sample_ratio = 50
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    train_loader, dev_loader, test_loader, dev_triple_ranges, raw_dev_set, test_triple_ranges, raw_test_set =\
        form_dataset(train_full_fn, dev_full_fn, test_full_fn, train_batch, test_batch, tokenizer, num_workers, sample_ratio)

    print(len(train_loader), len(dev_loader), len(test_loader))
    train_list = list(dev_loader)
    for batch in train_list:
        print(batch[0].shape)
        for ques, type_mask, att_mask, tag in zip(*batch):
            cur_len = int(att_mask.sum().item())
            assert(ques[cur_len:].sum().item() == PAD_TOKEN_ID*(len(ques)-cur_len))
            assert(type_mask[cur_len:].sum().item() == 1*(len(ques)-cur_len))
            print(tokenizer.decode(ques[:cur_len]))
            sent_len = int((type_mask==0).sum().item())
            print(tokenizer.decode(ques[sent_len:cur_len]))
            print(tag.item())
        break
