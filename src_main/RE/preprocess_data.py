import copy
import json
import pickle
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, RobertaTokenizer


def resolve_rel(rel):
    if rel[:19] != 'http://dbpedia.org/':
        assert(rel == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
        return {
            'rel_name': 'type',
            'rel_type': ''
        }
    rel = rel[19:]
    pos1 = rel.find('/')
    #pos2 = rel.find('/', pos1+1)
    return {
        'rel_type': rel[:pos1],
        'rel_name': rel[pos1+1:]
    }


def tokenize_and_convert_token_id(raw_fn, out_fn, tokenizer, is_test_set=False, re_data_aug=True, id_mask=None):
    # 预处理RE数据, 由于RE数据量大可节省训练脚本执行时间
    # 读入RE_Cand数据, 生成五元组, 需注意:
    #   1. 保证构建出的负例五元组中不包括正例
    #   2. 保证每个triple只有一个正例 (目前如此, 之后再看)
    #   3. Dbpedia有property/ontology的relation名称相同, 所以需把前缀也加上
    #   4. 可以支持paper中的数据增强, 将反向的数据加入正向作为负例, 注意仍需要保证1.!
    # 利用tokenizer转换token_id并pickle输出
    
    with open(raw_fn) as fin:
        raw_dat = json.load(fin)

    # 五元组cnt, 用以计算ranges
    quintuple_cnt = 0
    if is_test_set:
        ranges = []
        v_set = []
    toks, type_masks, tags = [], [], []
    
    collected_data_cnt = 0
    for dat in tqdm(raw_dat):
        if (id_mask is not None) and (dat['_id'] not in id_mask):
            continue
        collected_data_cnt += 1
        question = dat['Question']
        ques_toks = tokenizer.encode(question, add_special_tokens=False)    # 因为之后concat时统一加special tokens
        for triple, triple_rel_info in zip(dat['Sparql_struct'][0][1:], dat['Relation list']):
            # 当前triple的所有数据从哪开始
            pre_triple_cnt = quintuple_cnt
            gold = triple_rel_info['gold'][0]
            pos_cand = triple_rel_info['pos_cand']
            rev_cand = triple_rel_info['rev_cand']
            mention1 = triple[0][2]
            mention2 = triple[1][2]
            m1_toks = tokenizer.encode(mention1, add_special_tokens=False)
            m2_toks = tokenizer.encode(mention2, add_special_tokens=False)
            cur_sent_in_pos = [tokenizer.cls_token_id] + ques_toks + [tokenizer.sep_token_id] +\
                    m1_toks + [tokenizer.sep_token_id] + m2_toks + [tokenizer.sep_token_id]
            cur_sent_in_rev = [tokenizer.cls_token_id] + ques_toks + [tokenizer.sep_token_id] +\
                    m2_toks + [tokenizer.sep_token_id] + m1_toks + [tokenizer.sep_token_id]
            assert(len(cur_sent_in_pos) == len(cur_sent_in_rev))

            # 构建当前triple的pos/rev relation set
            pos_set, rev_set = set(), set()
            gold_rel_info = resolve_rel(gold)
            
            # 构建正向候选, 去重, 去除gold, 数据增强
            for rel in pos_cand:
                rel_info = resolve_rel(rel)
                pos_set.add((rel_info['rel_type'], rel_info['rel_name']))
                if re_data_aug:
                    # re数据增强: 将正向的候选也作为反向的负例, 让模型更好地学习mention顺序的影响
                    rev_set.add((rel_info['rel_type'], rel_info['rel_name']))
            
            # 构建反向候选, 去重, 因为gold都是pos所以无需删除gold
            # 但是数据增强时需要将gold加入
            if re_data_aug:
                rev_set.add((gold_rel_info['rel_type'], gold_rel_info['rel_name']))
            for rel in rev_cand:
                rel_info = resolve_rel(rel)
                rev_set.add((rel_info['rel_type'], rel_info['rel_name']))
                if re_data_aug:
                    # re数据增强: 将正向的候选也作为反向的负例, 让模型更好地学习mention顺序的影响
                    pos_set.add((rel_info['rel_type'], rel_info['rel_name']))
            
            # 最后再去除pos_set里面的gold, 避免rev加数据增强时又引入了; 这里remove就没问题了
            if (gold_rel_info['rel_type'], gold_rel_info['rel_name']) in pos_set:
                pos_set.remove((gold_rel_info['rel_type'], gold_rel_info['rel_name']))
            
            # 构建训练数据
            # gold
            rel = gold_rel_info['rel_type'] + ' ' + gold_rel_info['rel_name']
            rel_toks = tokenizer.encode(rel, add_special_tokens=False)
            cur_rel_in = rel_toks + [tokenizer.sep_token_id]
            cur_tok = copy.deepcopy(cur_sent_in_pos) + copy.deepcopy(cur_rel_in)
            cur_type_mask = [0]*len(cur_sent_in_pos) + [1]*len(cur_rel_in)
            assert(len(cur_tok) == len(cur_type_mask))
            toks.append(cur_tok)
            type_masks.append(cur_type_mask)
            tags.append(1)
            quintuple_cnt += 1
            if is_test_set:
                v_set.append([mention1, mention2, question, rel, '1'])
            # positive
            for rel_info in pos_set:
                rel = rel_info[0] + ' ' + rel_info[1]
                rel_toks = tokenizer.encode(rel, add_special_tokens=False)
                cur_rel_in = rel_toks + [tokenizer.sep_token_id]
                cur_tok = copy.deepcopy(cur_sent_in_pos) + copy.deepcopy(cur_rel_in)
                cur_type_mask = [0]*len(cur_sent_in_pos) + [1]*len(cur_rel_in)
                toks.append(cur_tok)
                type_masks.append(cur_type_mask)
                tags.append(0)
                quintuple_cnt += 1
                if is_test_set:
                    v_set.append([mention1, mention2, question, rel, '0'])
            for rel_info in rev_set:
                rel = rel_info[0] + ' ' + rel_info[1]
                rel_toks = tokenizer.encode(rel, add_special_tokens=False)
                cur_rel_in = rel_toks + [tokenizer.sep_token_id]
                cur_tok = copy.deepcopy(cur_sent_in_rev) + copy.deepcopy(cur_rel_in)
                cur_type_mask = [0]*len(cur_sent_in_rev) + [1]*len(cur_rel_in)
                toks.append(cur_tok)
                type_masks.append(cur_type_mask)
                tags.append(0)
                quintuple_cnt += 1
                if is_test_set:
                    v_set.append([mention2, mention1, question, rel, '0'])
            assert(len(toks) == len(type_masks) == len(tags) == quintuple_cnt)
            
            # 构建ranges
            if is_test_set:
                ranges.append([pre_triple_cnt, quintuple_cnt-1])
    
    # output results
    with open(out_fn, 'wb') as fout:
        if is_test_set:
            pickle.dump(([toks, type_masks, tags], ranges, v_set), fout)
        else:
            pickle.dump((toks, type_masks, tags), fout)
    assert(id_mask is None or len(id_mask) == collected_data_cnt), "id_mask contains unexisting data in raw data"
    print(collected_data_cnt)


def preprocess_with_dev_set():
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)
        train_set, dev_set = split_dat['train'], split_dat['dev']
    
    train_raw_fn = '../data/RE_gold/train_re_1hop.json'
    train_out_fn = '../data/RE_gold/Train_RE_gold_preprocessed_1hop_split.json'
    dev_out_fn = '../data/RE_gold/Dev_RE_gold_preprocessed_1hop_split.json'
    test_raw_fn = '../data/RE_gold/test_re_1hop.json'
    test_out_fn = '../data/RE_gold/Test_RE_gold_preprocessed_1hop_split.json'
    
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    
    # 是否使用RE数据增强生成训练数据?
    re_data_aug = True

    tokenize_and_convert_token_id(train_raw_fn, train_out_fn, tokenizer, is_test_set=False, re_data_aug=re_data_aug, id_mask=train_set)
    tokenize_and_convert_token_id(train_raw_fn, dev_out_fn, tokenizer, is_test_set=True, re_data_aug=re_data_aug, id_mask=dev_set)
    tokenize_and_convert_token_id(test_raw_fn, test_out_fn, tokenizer, is_test_set=True, re_data_aug=re_data_aug)


if __name__ == '__main__':
    preprocess_with_dev_set()
    exit(0)

    train_raw_fn = '../data/RE_gold/train_re_1hop.json'
    train_out_fn = '../data/RE_gold/Train_RE_gold_preprocessed_1hopdup.json'
    test_raw_fn = '../data/RE_gold/test_re_1hop.json'
    test_out_fn = '../data/RE_gold/Test_RE_gold_preprocessed_1hopdup.json'
    
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    
    # 是否使用RE数据增强生成训练数据?
    re_data_aug = True

    tokenize_and_convert_token_id(train_raw_fn, train_out_fn, tokenizer, is_test_set=False, re_data_aug=re_data_aug)
    tokenize_and_convert_token_id(test_raw_fn, test_out_fn, tokenizer, is_test_set=True, re_data_aug=re_data_aug)
