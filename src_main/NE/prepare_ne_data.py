# 预处理Node Extraction数据，生成字粒度标注序列
# 可配置生成: 有类别(E/V/T/VT) / 无类别(仅BIO) 的标签序列

import json
import sys
sys.path.append('../')
from transformers import DebertaV2Tokenizer, RobertaTokenizer
from tqdm import tqdm

from annotation.preprocess import parse_sparql
from QG.util import qg_tag2id


# tokenizer标志空格的标志
TOKENIZER_START_SIGN = 'Ġ'  # '▁' for deberta


def check_mention_identity(token_l, token_r, tokens, raw_mention, dat):
    # 比较tokenize后的token序列与原始char mention
    assert(token_l <= token_r)
    cur_mention = ''.join([ct.replace(TOKENIZER_START_SIGN, ' ') for ct in tokens[token_l:token_r+1]]).strip().replace(')?', ')').replace(').', ')')
    if cur_mention[-1] == ',':
        cur_mention = cur_mention[:-1]
    if cur_mention[-1] == '?':
        cur_mention = cur_mention[:-1]
    if cur_mention != raw_mention:
        print(token_l, '\t', token_r, '\t', dat['_id'], '\n', cur_mention, '\n', raw_mention, sep='', end='\n\n')


def has_token_span_overlap(token_l, token_r, gold_nodes):
    for cur_l, cur_r in gold_nodes.keys():
        if (token_l, token_r) == (cur_l, cur_r):
            return (token_l, token_r)
        elif token_l <= cur_r and token_r >= cur_l:
            return (cur_l, cur_r)
    return -1


def form_NEQG_gold_data(src, dst_ne, dst_qg, tokenizer, add_tag_type=False):
    # 根据src文件中的char级别l/r_pos信息, 运行tokenize转换成token级别位置
    # 再根据这些位置生成NE标签序列, 支持重叠标签的生成(VT, VE...)
    # TODO: QG的 <pointer, type> 序列实际上也可以一块生成
    
    with open(src) as fin:
        annotated_data = json.load(fin)
    type_relations = set()      # 记录除#type以外仍被认为是type (mention重叠) 的relation
    ne_output_data = []        # 记录token和标签序列, 即NE训练数据
    qg_output_data = []        # 记录node <type, ptr> 序列, 即QG训练数据

    for dat in (annotated_data):
        tokens = tokenizer.tokenize(dat['question'])
        # 先通过find计算token在question中的位置对应关系, 然后反过来计算char到token的映射
        char2token_pos, token_start_pos = dict(), []
        pos = 0
        for ix, token in enumerate(tokens):
            token_mention = token.replace(TOKENIZER_START_SIGN, '')
            if len(token_mention) == 0:     # 空token直接跳过
                continue
            l_pos = dat['question'].find(token_mention, pos)
            r_pos = l_pos + len(token_mention) - 1
            assert(l_pos != -1)
            assert(dat['question'][l_pos:r_pos+1].strip() == dat['question'][l_pos:r_pos+1])
            pos = r_pos + 1

            # 从l_pos到r_pos都映射到ix号token
            for cur_pos in range(l_pos, r_pos+1):
                assert((cur_pos) not in char2token_pos.keys())
                char2token_pos[cur_pos] = ix
            # 为sanity check, 也保存token开始的位置
            token_start_pos.append(l_pos)
        
        # 计算char级别位置到token位置的映射
        node_set = dat['node_set']
        # 在标签中会出现的node列表, 处理overlap, 包含node type
        gold_nodes = dict()
        # 处理NE数据时不重复计算同一个node
        already_annotated = set()
        # QG的gold数据, 包含node_sequence, 每个node的start/end token-pos
        # 注意需要单独添加target节点, 其中ASK类target规定为 <None, Variable>
        qg_node_sequence = []

        for triple in dat['triples']:
            head, tail = triple[0], triple[2]
            # 确定head和tail的类别, 考虑type类型
            head_type = 'variable' if '?' in head else 'entity'
            tail_type = 'variable' if '?' in tail else ('type' if ('type' in triple[1].lower()) else 'entity')
            if head_type != 'variable' and tail_type == 'type':
                tail_type = 'entity'
            # 对于NASA的几条数据特判一下
            if tail == 'http://dbpedia.org/resource/NASA':
                tail_type = 'entity'
            head_l, head_r = node_set[head]['l_pos'], node_set[head]['r_pos']
            tail_l, tail_r = node_set[tail]['l_pos'], node_set[tail]['r_pos']
            
            # 这里假设所有char级别l/r都不是空格 (因为之前算char2token_pos没考虑未被token包括的char pos)
            if head_l == -1:
                assert(head_r == -1 and node_set[head]['mention'] == 'None')
            if tail_l == -1:
                assert(tail_r == -1 and node_set[tail]['mention'] == 'None')
            
            # sanity checks
            if head_l not in token_start_pos and head_l != -1:
                print(1)
                print(head_l, tokens, dat['question'], head, node_set[head]['mention'])
            if tail_l not in token_start_pos and tail_l != -1:
                print(2)
                print(tail_l, tokens, dat['question'], tail, node_set[tail]['mention'])
            #if head_r in token_start_pos:
            #    print(3)
            #    print(head_r, tokens, dat['question'], head, node_set[head]['mention'])
            #if tail_r in token_start_pos:
            #    print(4)
            #    print(tail_r, tokens, dat['question'], tail, node_set[tail]['mention'])
            
            # 构建标注区间-head
            if head_l != -1 and (head not in already_annotated):
                already_annotated.add(head)
                head_token_l, head_token_r = char2token_pos[head_l], char2token_pos[head_r]
                # 检查tokens和原始char级别mention是否一致
                #check_mention_identity(head_token_l, head_token_r, tokens, node_set[head]['mention'], dat)
                res = has_token_span_overlap(head_token_l, head_token_r, gold_nodes)
                if res == -1:
                    gold_nodes[(head_token_l, head_token_r)] = {head_type}
                elif res == (head_token_l, head_token_r):
                    print('Overlap head: ', dat['_id'], '\t', dat['question'], '\n', triple[0], dat['triples'], sep='', end='\n\n')
                    gold_nodes[(head_token_l, head_token_r)].add(head_type)
                else:
                    # 少量case出现部分重叠, 由于我们的序列标注模型无法处理重叠序列, 这里仅保留更长的区间
                    print('Partial overlap head: ', dat['_id'], '\t', dat['question'], '\n', triple[0], '\n', tokens, '\n', (head_token_l, head_token_r), '\t', res, sep='', end='\n\n')
                    if head_token_r-head_token_l > res[1]-res[0]:
                        print('Replace the old span')
                        del gold_nodes[res]
                        gold_nodes[(head_token_l, head_token_r)] = {head_type}
            
            # 在 QG sequence 中的 head node
            if head_l == -1:
                head_token_l, head_token_r = -1, -1
            else:
                head_token_l, head_token_r = char2token_pos[head_l], char2token_pos[head_r]
            qg_head = {
                "tag": head_type,   # head-type没啥异议, 就entity和variable两类, 直接分即可
                "start": head_token_l,
                "end": head_token_r,
                "sparql_tok": head,
            }
            
            # tail
            if tail_l != -1 and (tail not in already_annotated):
                already_annotated.add(tail)
                tail_token_l, tail_token_r = char2token_pos[tail_l], char2token_pos[tail_r]
                #check_mention_identity(tail_token_l, tail_token_r, tokens, node_set[tail]['mention'], dat)
                res = has_token_span_overlap(tail_token_l, tail_token_r, gold_nodes)
                if res == -1:
                    gold_nodes[(tail_token_l, tail_token_r)] = {tail_type}
                elif res == (tail_token_l, tail_token_r):
                    if tail_type != 'type':
                        type_relations.add(triple[1])
                        #print('Non-type overlap tail: ', dat['_id'], '\t', dat['question'], '\n', triple[2], sep='', end='\n\n')
                    if gold_nodes[(tail_token_l, tail_token_r)] != {'variable'}:
                        print('Not variable-type: ', dat['_id'], '\t', gold_nodes[(tail_token_l, tail_token_r)], '\t', dat['question'], '\n', triple[2], sep='', end='\n\n')
                    # NOTE: 这里直接强制当做variable-type了!
                    gold_nodes[(tail_token_l, tail_token_r)].add('type')
                    # NOTE: 这里不管mention overlap的是entity还是variable, 都转成了type
                    # 强转variable是为了消除自环便于训练, 这一做法值得商榷, 但是影响面只有~3条, 估计差别不大
                    tail_type = 'type'
                else:
                    print('Partial overlap tail: ', dat['_id'], '\t', dat['question'], '\n', triple[2], '\n', tokens, '\n', (tail_token_l, tail_token_r), '\t', res, sep='', end='\n\n')
                    if tail_token_r-tail_token_l > res[1]-res[0]:
                        print('Replace the old span\n')
                        del gold_nodes[res]
                        gold_nodes[(tail_token_l, tail_token_r)] = {tail_type}
            
            # 在 QG sequence 中的 tail node
            if tail_l == -1:
                tail_token_l, tail_token_r = -1, -1
            else:
                tail_token_l, tail_token_r = char2token_pos[tail_l], char2token_pos[tail_r]
            qg_tail = {
                "tag": tail_type,   # tail-type 在上面处理了, 部分强转成type
                "start": tail_token_l,
                "end": tail_token_r,
                "sparql_tok": tail,
            }

            # 将当前triple (node pair) 加入 qg_sequence
            qg_node_sequence.append([qg_head, qg_tail])

        # 到这里, gold_nodes计算完成, 根据它生成NE标签
        cur_tags = ['O'] * len(tokens)
        for (node_l, node_r), node_type in gold_nodes.items():
            if node_type == {'variable', 'type'}:
                tag_type = 'VT'
            elif node_type == {'type'}:
                tag_type = 'CT'
            elif node_type == {'entity'}:
                tag_type = 'E'
            elif node_type == {'variable'}:
                tag_type = 'V'
            else:
                assert(False), node_type
            # 输出 BIO 标签序列
            cur_tags[node_l] = 'B_' + tag_type if add_tag_type else 'B'
            for token_ix in range(node_l+1, node_r+1):
                cur_tags[token_ix] = 'I_' + tag_type if add_tag_type else 'I'
        
        cur_output_data = {
            '_id': dat['_id'],
            'tokens': tokens,
            'tag_seq': cur_tags
        }
        ne_output_data.append(cur_output_data)

        # QG规范化: 将triple按 mention pointer (start token ix) 顺序排序, 先head后tail
        qg_node_sequence.sort(key=lambda x:10000*x[0]['start']+x[1]['start'])
        # triple的head和tail也按照 <start, tag_type_id> 从小到大, 因为QG是不考虑head-tail顺序的
        qg_node_sequence = [sorted(tri, key=lambda x:10000*x['start'] + qg_tag2id[x['tag']]) for tri in qg_node_sequence]

        # 扁平化, 直接弄成node sequence即可
        qg_node_sequence = sum(qg_node_sequence, [])

        # 加入 qg_sequence 的 target
        raw_spq = dat['raw_sparql']
        res = parse_sparql(raw_spq)
        if res['type'] != 'ask':
            target_node = res['target']
            assert('?' in target_node)
            target_l, target_r = node_set[target_node]['l_pos'], node_set[target_node]['r_pos']
            if target_l == -1:
                assert(target_r == -1)
                target_token_l, target_token_r = -1, -1
            else:
                target_token_l, target_token_r = char2token_pos[target_l], char2token_pos[target_r]
            qg_target = {
                "tag": 'variable',
                "start": target_token_l,
                "end": target_token_r,
                "sparql_tok": target_node,
            }
        else:
            qg_target = {
                "tag": 'variable',
                "start": -1,
                "end": -1,
                "sparql_tok": '?ASK_target',
            }
        qg_node_sequence.insert(0, qg_target)
        
        cur_output_data = {
            '_id': dat['_id'],
            'tokens': tokens,
            'node_seq': qg_node_sequence
        }
        qg_output_data.append(cur_output_data)

    print('Type-like relations:\n', type_relations, '\n', sep='')
    with open(dst_ne, 'w') as fout:
        json.dump(ne_output_data, fout, indent=4)
    with open(dst_qg, 'w') as fout:
        json.dump(qg_output_data, fout, indent=4)


def main():
    train_src = '../annotation/post_processed/Train_full_data_annotated.json'
    train_dst_ne = '../data/NEQG_gold/Train_NE_gold.json'
    train_dst_qg = '../data/NEQG_gold/Train_QG_gold.json'
    test_src = '../annotation/post_processed/Test_full_data_annotated.json'
    test_dst_ne = '../data/NEQG_gold/Test_NE_gold.json'
    test_dst_qg = '../data/NEQG_gold/Test_QG_gold.json'
    add_tag_type = False

    global TOKENIZER_START_SIGN
    #tokenizer_dir = '../data/pretrained/deberta_xlarge_tokenizer/'
    #tokenizer_class = DebertaV2Tokenizer
    #TOKENIZER_START_SIGN = '▁'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    TOKENIZER_START_SIGN = 'Ġ'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    
    form_NEQG_gold_data(train_src, train_dst_ne, train_dst_qg, tokenizer, add_tag_type=add_tag_type)
    print('====================================================')
    form_NEQG_gold_data(test_src, test_dst_ne, test_dst_qg, tokenizer, add_tag_type=add_tag_type)


if __name__ == '__main__':
    main()