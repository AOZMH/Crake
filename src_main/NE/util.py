# 利用prepare_ne_data预处理好的数据, 生成NE模型dataloader

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, RobertaTokenizer


# tag dict for Node-Extraction
tags = ('O', 'B_E', 'I_E', 'B_V', 'I_V', 'B_VT', 'I_VT', 'B_CT', 'I_CT')
#tags = ('O', 'B', 'I')
tag2id = {tag:tid for tag,tid in zip(tags, range(len(tags)))}
id2tag = {tid:tag for tag,tid in tag2id.items()}

# Tokenizer Info
PAD_TOKEN_ID = None


def convert_tokens_to_ids_full(train_full_fn, test_full_fn, tokenizer, train_dev_split):
    with open(train_full_fn) as train_fin, open(test_full_fn) as test_fin:
        train_full = json.load(train_fin)
        test_full = json.load(test_fin)
    # Split train-dev
    train_set, dev_set = train_dev_split['train'], train_dev_split['dev']
    
    # Process train & test set
    train_data = [[], []]
    dev_data = [[], []]
    test_data = [[], []]
    for ix, dat in enumerate(train_full + test_full):
        tokens, tags = dat['tokens'], dat['tag_seq']
        token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        tag_ids = [0] + [tag2id[cur_tag] for cur_tag in tags] + [0]
        assert(len(tag_ids)==len(token_ids) and len(token_ids)==len(tokens)+2)
        if ix < len(train_full):
            if dat['_id'] in dev_set:
                dev_data[0].append(token_ids)
                dev_data[1].append(tag_ids)
            else:
                train_data[0].append(token_ids)
                train_data[1].append(tag_ids)
        else:
            test_data[0].append(token_ids)
            test_data[1].append(tag_ids)
    assert(len(train_data[0]) == len(train_set) and len(dev_data[0]) == len(dev_set)), "train_dev_split contains unexisting data in raw data"
    
    return train_data, dev_data, test_data


class NEDataset(Dataset):
    def __init__(self, all_data):
        assert(len(all_data[0]) == len(all_data[1]))
        self.ques = all_data[0]
        self.tags = all_data[1]
    
    def __len__(self):
        return len(self.ques)
    
    def __getitem__(self, idx):
        return self.ques[idx], self.tags[idx]


def NE_collate(batch):
    # batch = [dataset[ix1], dataset[ix2], ...] = [[ques_i, tags_i], ...]
    lst = []
    padded_seq = []
    att_mask = []
    ne_tags = []

    max_seq_len = len(max(batch, key=lambda x:len(x[0]))[0])
    for seq, tag in batch:
        assert(len(seq)==len(tag))
        padded_seq.append(seq + [PAD_TOKEN_ID]*(max_seq_len-len(seq)))
        att_mask.append([1]*len(seq) + [0]*(max_seq_len-len(seq)))
        ne_tags.append(tag + [0]*(max_seq_len-len(seq)))

    lst.append(torch.LongTensor(padded_seq))
    lst.append(torch.FloatTensor(att_mask))
    lst.append(torch.LongTensor(ne_tags))

    return lst


def form_dataset(train_full_fn, test_full_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, train_dev_split):
    # NE train test dataloader
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = tokenizer.pad_token_id
    print('Tokenizer loaded.')
    train_data, dev_data, test_data = convert_tokens_to_ids_full(train_full_fn, test_full_fn, tokenizer, train_dev_split)
    print('Tokens converted to ids.\nTrain / Dev / Test length: {} / {} / {}.'.format(len(train_data[0]), len(dev_data[0]), len(test_data[0])))
    train_dataset, dev_dataset, test_dataset = NEDataset(train_data), NEDataset(dev_data), NEDataset(test_data)
    
    # dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, collate_fn=NE_collate, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=test_batch, collate_fn=NE_collate, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, collate_fn=NE_collate, shuffle=False)
    print('Dataloaders formed.')
    return train_loader, dev_loader, test_loader


if __name__ == '__main__':
    train_full_fn = '../data/NEQG_gold/Train_NE_gold_w_tag.json'
    test_full_fn = '../data/NEQG_gold/Test_NE_gold_w_tag.json'

    #tokenizer_dir = '../data/pretrained/deberta_xlarge_tokenizer/'
    #tokenizer_class = DebertaV2Tokenizer
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer

    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)

    train_batch, test_batch = 4, 4
    train_loader, dev_loader, test_loader = form_dataset(train_full_fn, test_full_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, split_dat)
    
    print(len(train_loader), len(dev_loader), len(test_loader))
    train_list = list(dev_loader)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    for batch in train_list:
        print(batch[0].shape)
        for ques, mask, tag in zip(*batch):
            cur_len = int(mask.sum().item())
            assert(ques[cur_len:].sum().item() == PAD_TOKEN_ID*(len(ques)-cur_len))
            assert(tag[cur_len:].sum().item() == 0)
            print(tokenizer.convert_ids_to_tokens(ques[:cur_len]))
            print([id2tag[cur_tag.item()] for cur_tag in tag[:cur_len]])
        break
