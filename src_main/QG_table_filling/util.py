# Utils for table-filling-based QG module
# 1. Convert table-filling data from original decoder-based preprocessed data
# 2. Connect special edges (<s>-target, </s>=None...)
# 3. Perform table paddings to form dataloaders

import sys
sys.path.append('../')
import json
from itertools import product
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, RobertaTokenizer

from NE.util import convert_tokens_to_ids_full as convert_tokens_to_ids_full_ne
from NE.util import id2tag as ne_id2tag

qg_tags = ('<start>', '<end>', 'entity', 'variable', 'type')
qg_tag2id = {tag:tid for tag,tid in zip(qg_tags, range(len(qg_tags)))}
qg_id2tag = {tid:tag for tag,tid in qg_tag2id.items()}

# Tokenizer Info
PAD_TOKEN_ID = None


def convert_tokens_to_ids_and_form_adj_matrix(train_full_fn, test_full_fn, tokenizer, train_dev_split):
    # 1. 与QG一样, 转换token_iod, 增加特殊token并修正offset
    # 2. 构造token邻接矩阵, 考虑<s>和</s>, 但不考虑padding (再collate时才能进行padding)
    with open(train_full_fn) as train_fin, open(test_full_fn) as test_fin:
        train_full = json.load(train_fin)
        test_full = json.load(test_fin)
    # Split train-dev
    train_set, dev_set = train_dev_split['train'], train_dev_split['dev']

    # Process train & test set
    train_data = [[], [], [], [], []]   # 分别存token_ids, node_types, node_ptrs, node_ends, adj_matrix
    dev_data = [[], [], [], [], []]
    test_data = [[], [], [], [], []]
    for dat_ix, dat in enumerate(train_full + test_full):
        tokens, node_seq = dat['tokens'], dat['node_seq']
        # 加上[CLS]和[SEP]
        token_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        # Tag sequence 要喂进decoder, 加上start和end
        node_types = [qg_tag2id['<start>']] + [qg_tag2id[node['tag']] for node in node_seq] + [qg_tag2id['<end>']]
        # ptrs代表当前输入decoder的node的pointer, 所以只需要到最后一个原始node即可, 无需包括<end>
        # 另外由于原句加入了[CLS]/<s>, 这里offset加了1; 下面node_ends同理
        node_ptrs = [0] + [node['start']+1 for node in node_seq]
        # span ends 仅在NEQG评估时恢复gold span时用到
        node_ends = [node['end']+1 for node in node_seq]

        # 构造token级别邻接表, 包括上面加入的[CLS]和[SEP]位置, 注意此时未考虑padding
        # NOTE: 结点图是无向图, 所以每条边要在对称的元素上都标为1
        adj_matrix = [[0 for i in range(len(token_ids))] for j in range(len(token_ids))]
        
        # 这里规定所有None mention(ptr=-1)对应到句末的特殊token, 即[SEP], </s>等
        # 一般情况下+1是考虑了<s>, 原token是从1开始的; 0和(len(token_ids)-1)分别对应<s>和</s>
        format_ptr = lambda ptr : (ptr + 1) if ptr != -1 else (len(token_ids) - 1)

        # 处理target结点
        target_start, target_end = format_ptr(node_seq[0]['start']), format_ptr(node_seq[0]['end'])
        for target_idx in range(target_start, target_end + 1):
            adj_matrix[0][target_idx] = 1
            adj_matrix[target_idx][0] = 1
        
        # 处理其他结点, 注意两个span中每对token间都连边
        for ix in range(1, len(node_seq), 2):
            head, tail = node_seq[ix], node_seq[ix + 1]
            # TODO consider None mention, cannot use product! -> 没事, 直接对应到</s>了, 也可以认为就是这个span
            # TODO consider changed offset -> format_none_ptr这个函数处理了<s>的offset更新
            head_s, head_e, tail_s, tail_e = map(format_ptr, (head['start'], head['end'], tail['start'], tail['end']))
            for i_token, j_token in product(range(head_s, head_e + 1), range(tail_s, tail_e + 1)):
                # 1. 注意这里已经考虑了<s>带来的offset后移
                # 2. 对于None, format_ptr已经将其对应到</s>了, 也可以一致处理
                adj_matrix[i_token][j_token] = 1
                adj_matrix[j_token][i_token] = 1

        if dat_ix < len(train_full):
            if dat['_id'] in dev_set:
                dev_data[0].append(token_ids)
                dev_data[1].append(node_types)
                dev_data[2].append(node_ptrs)
                dev_data[3].append(node_ends)
                dev_data[4].append(adj_matrix)
            else:
                train_data[0].append(token_ids)
                train_data[1].append(node_types)
                train_data[2].append(node_ptrs)
                train_data[3].append(node_ends)
                train_data[4].append(adj_matrix)
        else:
            test_data[0].append(token_ids)
            test_data[1].append(node_types)
            test_data[2].append(node_ptrs)
            test_data[3].append(node_ends)
            test_data[4].append(adj_matrix)
    return train_data, dev_data, test_data


class QG_TF_Dataset(Dataset):
    # QG question-node_matrix 数据类型
    # 也可包括 NE 序列标注数据类型, 用于NEQG联训
    def __init__(self, all_data):
        for dat in all_data:
            assert(len(dat) == len(all_data[0]))
        self.ques = all_data[0]
        # 结构见 **QG_TF_collate** !
        self.node_types = all_data[1]
        self.node_ptrs = all_data[2]
        self.node_ends = all_data[3]
        self.adj_matrix = all_data[4]
        if len(all_data) > 5:
            self.ne_tags = all_data[5]

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        if hasattr(self, 'ne_tags'):
            return self.ques[idx], self.ne_tags[idx], self.node_types[idx], self.node_ptrs[idx], self.node_ends[idx], self.adj_matrix[idx]
        else:
            return self.ques[idx], self.node_types[idx], self.node_ptrs[idx], self.node_ends[idx], self.adj_matrix[idx]


def QG_TF_collate(batch):
    # batch = [dataset[ix1], dataset[ix2], ...] = [[ques_i, tags_i], ...]
    lst = []
    padded_seq = []
    att_mask = []
    padded_query, padded_ptrs, node_ends = [], [], []
    if len(batch[0]) == 6:
        ne_tags = []
    query_end_id = qg_tag2id['<end>']
    # 新增: token级别padding后的邻接矩阵, <batch, max_seq_len, max_seq_len>
    adj_matrices = []
    matrix_masks = []

    max_seq_len = len(max(batch, key=lambda x:len(x[0]))[0])
    # 取node_end, 这里是倒数第二个; 同时注意query长度应+2
    max_que_len = len(max(batch, key=lambda x:len(x[-2]))[-2]) + 2
    for cur_item in batch:
        if len(cur_item) == 5:  # pure qg
            ques, node_type, node_ptr, node_end, adj_matrix = cur_item
        else:   # ne & qg
            #raise NotImplementedError
            ques, ne_tag, node_type, node_ptr, node_end, adj_matrix = cur_item
            assert(len(ques)==len(ne_tag))
            ne_tags.append(ne_tag + [0]*(max_seq_len-len(ne_tag)))

        # 保留了decoder QG的数据, 方便恢复/验证gold数据; 具体结构同QG下的util.py
        assert(len(node_type) == (len(node_ptr)+1) == (len(node_end)+2))
        # for encoder
        padded_seq.append(ques + [PAD_TOKEN_ID]*(max_seq_len-len(ques)))
        att_mask.append([1]*len(ques) + [0]*(max_seq_len-len(ques)))
        # for decoder
        padded_query.append(node_type + [query_end_id]*(max_que_len-len(node_type)))
        padded_ptrs.append(node_ptr + [0]*(max_que_len-len(node_ptr)-1))
        node_ends.append(node_end)
        # 邻接矩阵padding
        adj_matrix_cp = [adj_matrix_row + [0]*(max_seq_len-len(ques)) for adj_matrix_row in adj_matrix] +\
                        [[0] * max_seq_len for __i__ in range(max_seq_len-len(ques))]
        #for ix in range(len(adj_matrix)):
        #    assert(len(adj_matrix[ix]) == len(ques))
        #    adj_matrix[ix] += [0]*(max_seq_len-len(ques))
        #for ix in range(max_seq_len-len(ques)):
        #    adj_matrix.append([0]*max_seq_len)
        adj_matrices.append(adj_matrix_cp)
        matrix_mask = [[1] * len(ques) + [0] * (max_seq_len - len(ques))] * len(ques)
        matrix_mask += [[0] * max_seq_len] * (max_seq_len - len(ques))
        matrix_masks.append(matrix_mask)

    lst.append(torch.LongTensor(padded_seq))
    lst.append(torch.FloatTensor(att_mask))
    if len(batch[0]) == 6:
        lst.append(torch.LongTensor(ne_tags))
    lst.append(torch.LongTensor(adj_matrices))
    lst.append(torch.BoolTensor(matrix_masks))
    lst.append(torch.LongTensor(padded_query))
    lst.append(torch.LongTensor(padded_ptrs))
    lst.append(node_ends)

    return lst


def form_table_filling_dataset(train_qg_fn, test_qg_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, train_dev_split, train_ne_fn=None, test_ne_fn=None):
    # 生成QG-table-filling的train和test dataloader
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = tokenizer.pad_token_id
    print('Tokenizer loaded.')

    train_data_qg, dev_data_qg, test_data_qg = convert_tokens_to_ids_and_form_adj_matrix(train_qg_fn, test_qg_fn, tokenizer, train_dev_split)
    if train_ne_fn and test_ne_fn:
        print('Also loading NE data from {}, {}'.format(train_ne_fn, test_ne_fn))
        #raise NotImplementedError('NEQG loader yet supported...')
        train_data_ne, dev_data_ne, test_data_ne = convert_tokens_to_ids_full_ne(train_ne_fn, test_ne_fn, tokenizer, train_dev_split)
        # 将ne_tags数据并入data_qg中
        train_data_qg.append(train_data_ne[1])
        dev_data_qg.append(dev_data_ne[1])
        test_data_qg.append(test_data_ne[1])
    print('Tokens converted to ids, Adjacent matrix formed.\nTrain / Dev / Test length: {} / {} / {}.'.format(len(train_data_qg[0]), len(dev_data_qg[0]), len(test_data_qg[0])))
    train_dataset, dev_dataset, test_dataset = QG_TF_Dataset(train_data_qg), QG_TF_Dataset(dev_data_qg), QG_TF_Dataset(test_data_qg)

    # dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, collate_fn=QG_TF_collate, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=test_batch, collate_fn=QG_TF_collate, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, collate_fn=QG_TF_collate, shuffle=False)
    print('Dataloaders formed.')
    return train_loader, dev_loader, test_loader


if __name__ == '__main__':
    train_qg_fn = '../data/NEQG_gold/Train_QG_gold.json'
    test_qg_fn = '../data/NEQG_gold/Test_QG_gold.json'
    train_ne_fn = '../data/NEQG_gold/Train_NE_gold_w_tag.json'
    test_ne_fn = '../data/NEQG_gold/Test_NE_gold_w_tag.json'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer

    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)

    train_batch, test_batch = 4, 4
    train_loader, dev_loader, test_loader = form_table_filling_dataset(train_qg_fn, test_qg_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, split_dat, train_ne_fn, test_ne_fn)
    
    print(len(train_loader), len(dev_loader), len(test_loader))
    train_list = list(train_loader)+list(dev_loader)+list(test_loader)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    for batch in train_list:
        #print(batch[0].shape)
        for ques, mask, ne_tag, adj_matrix, matrix_mask, node_types, node_ptrs, node_ends in zip(*batch):
            cur_len_enc = int(mask.sum().item())
            assert(ques[cur_len_enc:].sum().item() == PAD_TOKEN_ID*(len(ques)-cur_len_enc))
            tokens = tokenizer.convert_ids_to_tokens(ques[:cur_len_enc])
            
            cur_len_dec = len(node_ends)
            assert(node_types[0].item()==qg_tag2id['<start>'] and node_types[cur_len_dec+1:].sum().item() == qg_tag2id['<end>']*(len(node_types)-cur_len_dec-1))
            assert(node_ptrs[0].item()==0 and node_ptrs[cur_len_dec+1:].sum().item() == 0)
            assert(len(node_types)==len(node_ptrs)+1)
            
            # 验证: 形状, 对称性, <s>唯一连边, padding部无边, 打印边示例看问题
            assert(len(ques) == len(adj_matrix) == len(adj_matrix[0]))
            assert( (adj_matrix.T == adj_matrix).sum() == len(adj_matrix)**2 )
            target_s, target_e = [], []
            for ix in range(len(adj_matrix[0]) - 1):
                if adj_matrix[0][ix] == 0 and adj_matrix[0][ix + 1] == 1:
                    target_s.append(ix + 1)
                if adj_matrix[0][ix] == 1 and adj_matrix[0][ix + 1] == 0:
                    target_e.append(ix)
            if adj_matrix[0][-1].item() == 1:
                target_e.append(len(adj_matrix) - 1)
            assert(len(target_s) == len(target_e) == int(target_s[0] <= target_e[0]) == 1)
            assert(adj_matrix[cur_len_enc:].sum() == adj_matrix[:, cur_len_enc:].sum() == 0)
            assert(adj_matrix.shape == matrix_mask.shape)
            for ix in range(len(matrix_mask)):
                for jx in range(len(matrix_mask)):
                    assert(matrix_mask[ix][jx].item() == int(ix < cur_len_enc and jx < cur_len_enc))
            
            # NE 验证
            assert(ne_tag[cur_len_enc:].sum().item() == 0)
            
            #continue
            # 打印几个sample的结点, 检查一下
            print('===============================\n' + ''.join(tokens[1:cur_len_enc-1]).replace('\u0120', ' '))
            print([ne_id2tag[tid.item()] for tid in ne_tag[1:cur_len_enc-1]])
            for tp, st, ed in zip(node_types[1:], node_ptrs[1:], node_ends):
                tp, st = tp.item(), st.item()
                assert(st <= ed)
                assert(st >= 0)
                print(qg_id2tag[tp], ''.join(tokens[st:ed+1]).replace('\u0120', ' '))
            for iix in range(len(adj_matrix)):
                for jix in range(iix + 1):
                    if adj_matrix[iix][jix].item():
                        print(tokens[iix], ' --> ', tokens[jix])
        break
    
