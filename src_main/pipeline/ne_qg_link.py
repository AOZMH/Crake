# See **eval.py** in NE/QG for more details
# Do NE+QG (and corresponding postprocessing) and entity linking in pipeline

import sys
sys.path.append('../')
import json
import copy
from itertools import product
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer

from QG.eval import match_ptr_to_node, struct_regularization, run_model_local, sort_node_seq_by_ptr, sort_node_seq_by_tag
from QG.util import form_dataset, qg_id2tag
from QG.model import Multitask_Model
from NE.eval import get_nodes
from NE.model import NE_Model
from Sent_CLS.classify_question import judge_ask, judge_count
from lookup_entity_linker import get_pkubase_client, dbpedia_lookup_entity_linker, link_entity, link_type


def load_models_and_tokenizers(params):
    # 导入params指定的模型(NE&QG)及tokenizer

    with open(params['model_config_fn']) as fin:
        model_config = json.load(fin)
    train_qg_fn = '../data/NEQG_gold/Train_QG_gold.json'
    test_qg_fn = '../data/NEQG_gold/Test_QG_gold.json'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    train_batch, test_batch = model_config['train_batch'], model_config['test_batch']
    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)

    # Dataloaders
    train_loader, dev_loader, test_loader = form_dataset(train_qg_fn, test_qg_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, split_dat)
    params['dev_loader'] = dev_loader
    params['test_loader'] = test_loader
    
    # Tokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    params['tokenizer'] = tokenizer
    
    # QG Model
    qg_model = Multitask_Model(model_config)
    output_model_file = model_config['checkpoint_route']
    print("Loading QG model from {}...".format(output_model_file))
    qg_model.load_state_dict(torch.load(output_model_file, map_location=torch.device('cpu')))
    # To GPU
    qg_device_ids = [int(gpu_id) for gpu_id in model_config.get('gpus', '0,1').split(',')]
    qg_gpu = qg_device_ids[0]
    qg_device = torch.device("cuda:{}".format(qg_gpu) if torch.cuda.is_available() else "cpu")
    qg_model = qg_model.to(qg_device)
    print("QG model running on:", qg_device)
    params['qg_model'] = qg_model
    
    # NE Model
    if 'ne_model_config_route' in model_config.keys():
        ne_model_config_fn = model_config['ne_model_config_route']
        print("Loading assisting NE model from {}...".format(ne_model_config_fn))
        with open(ne_model_config_fn) as fin:
            ne_model_config = json.load(fin)
        ne_model = NE_Model(ne_model_config)
        ne_model.load_state_dict(torch.load(ne_model_config['checkpoint_route'], map_location=torch.device('cpu')))
        print('Assisting NE model loaded from {}.'.format(ne_model_config['checkpoint_route']))
        # To GPU
        ne_device_ids = [int(gpu_id) for gpu_id in ne_model_config.get('gpus', '0,1').split(',')]
        ne_gpu = ne_device_ids[-1]
        ne_device = torch.device("cuda:{}".format(ne_gpu) if torch.cuda.is_available() else "cpu")
        ne_model = ne_model.to(ne_device)
        print("NE model running on:", ne_device)
    else:
        print('No assisting NE model specified.')
        assert(False)
        ne_model = None
    params['ne_model'] = ne_model

    # Entity linking (with cache)
    params['linker'] = dbpedia_lookup_entity_linker(cache_fn = '../data/lookup_cache/cache_new.dat')
    # KB client for entity linking (with cache)
    params['kb_client'] = get_pkubase_client(port=9275)
    # 链接需要用到train数据
    train_type_mapping_fn = '../data/mappings/train_type_mappings_ques.json'
    with open(train_type_mapping_fn) as fin:
        params['train_type_mappings'] = json.load(fin)
    
    return params


def get_raw_info():
    # Load sparqls & other info
    raw_train_fn = '../annotation/post_processed/Train_full_data_annotated.json'
    raw_test_fn = '../annotation/post_processed/Test_full_data_annotated.json'
    with open(raw_train_fn) as fin_train, open(raw_test_fn) as fin_test:
        train_dat = json.load(fin_train)
        test_dat = json.load(fin_test)
    info_data = {dat['_id']:{
        'Original sparql': dat['raw_sparql'],
        'Readable_sparql': dat['readable_sparql'],
    } for dat in train_dat + test_dat}

    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)
    dev_idx2global_id, cur_dev_idx = {}, 0
    for dat in train_dat:
        if dat['_id'] not in split_dat['dev']:
            continue
        dev_idx2global_id[cur_dev_idx] = dat['_id']
        cur_dev_idx += 1
    assert(cur_dev_idx == len(split_dat['dev']) == len(dev_idx2global_id))
    test_idx2global_id = {cur_idx: dat['_id'] for cur_idx, dat in enumerate(test_dat)}
    return info_data, dev_idx2global_id, test_idx2global_id


def nodes_to_triples(nodes):
    # node sequence: [[node_type_id, [l_pos, r_pos]], ...]
    # =>
    # triples: [head_node, [triple_1_head, triple_1_tail], ...]
    triples = [nodes[0]]
    for ix in range(1, len(nodes), 2):
        triples.append([nodes[ix], nodes[ix+1]])
    return triples


def node_info_convert(node, params):
    # [node_type_id, [l_pos, r_pos]]
    # =>
    # [sparql_node(str), node_type(str), node_mention(str)]
    # NOTE: Linking not yet performed, only a placeholder is filled
    tokenizer = params['tokenizer']
    sent, valid_len = params['sent'], params['valid_len']
    var_mention_format = lambda x:x.replace('\"','_').replace(' ', '_').replace('.', '_').replace(',', '_').replace('-', '_').replace(':', '_').replace('\'', '_').replace('\\', '_')
    node_mention = tokenizer.decode(sent[node[1][0]:node[1][1]+1]).strip()
    node_type = qg_id2tag[node[0]]
    # 考虑no-mention, 直接将其mention设置为None (与标注一致)
    if node[1][0] == node[1][1] == 0:
        node_mention = 'None'
    return [
        "?" + var_mention_format(node_mention),
        node_type,
        node_mention
    ]


def link_and_expand_triples(triples, params):
    # 对于entity/type进行linking; 将topk结果展开成多个sparql struct
    # NOTE: 为规范, 要求triples经过**sort_node_seq_by_ptr**排序!
    raw_ques = params['question']
    el_topk = params['el_topk']
    target_node = node_info_convert(triples[0], params)
    
    # 每个triple的所有可能均组成iter_list, 若没有E/T则是一个仅一个元素的list
    # 若有一个E/T则取top-k的link结果构成k中triple候选
    # 若有两个E/T则取所有召回结果进行排列组合
    iter_list = [[target_node]]     # target就这一种可能性
    has_entity, has_type = False, False
    for triple in triples[1:]:
        head_node, tail_node = node_info_convert(triple[0], params), node_info_convert(triple[1], params)
        if head_node[1] == 'variable' and tail_node[1] == 'variable':
            iter_list.append([[head_node, tail_node]])
            continue
        
        # 先获得head&tail的link结果
        params['mention'] = head_node[2]
        if head_node[1] == 'type':
            head_res = link_type(params)
            assert(len(head_res) in (0, 1))
            has_type = has_type or (len(head_res) > 0)
        elif head_node[1] == 'entity':
            head_res, _ = link_entity(params)
            has_entity = has_entity or (len(head_res) > 0)

        params['mention'] = tail_node[2]
        if tail_node[1] == 'type':
            tail_res = link_type(params)
            assert(len(tail_res) in (0, 1))
            has_type = has_type or (len(tail_res) > 0)
        elif tail_node[1] == 'entity':
            tail_res, _ = link_entity(params)
            has_entity = has_entity or (len(tail_res) > 0)
        
        # 利用link结果构建triple候选
        if head_node[1] == 'variable' and len(tail_res) > 0:    # 无link结果, 则舍弃这个triple
            triple_cands = []   # 可能有多个link结果, 对应多个triple候选
            for cur_ent in tail_res[:el_topk]:
                tail_node[0] = cur_ent['resource']
                triple_cands.append([head_node, copy.deepcopy(tail_node)])
            iter_list.append(triple_cands)

        elif tail_node[1] == 'variable' and len(head_res) > 0:  # 无link结果, 则舍弃这个triple
            triple_cands = []   # 可能有多个link结果, 对应多个triple候选
            for cur_ent in head_res[:el_topk]:
                head_node[0] = cur_ent['resource']
                triple_cands.append([copy.deepcopy(head_node), tail_node])
            iter_list.append(triple_cands)
        
        elif head_node[1] != 'variable' and tail_node[1] != 'variable' and len(head_res) > 0 and len(tail_res) > 0:   # 两个都是E/T, 取所有link候选
            assert(params['is_ask'])    # 否则, 在struct_regularization中应该将其去除了
            # 不用el-topk筛选, 而是保留所有候选
            triple_cands = []
            if params['ask_full_cand']:
                cands = product(head_res, tail_res)
            else:
                cands = product(head_res[:el_topk], tail_res[:el_topk])
            for head_ent, tail_ent in cands:
                head_node[0] = head_ent['resource']
                tail_node[0] = tail_ent['resource']
                triple_cands.append([copy.deepcopy(head_node), copy.deepcopy(tail_node)])
            iter_list.append(triple_cands)
    
    # 对于没有E/T的情况均舍弃. 这有可能由于: 1.本身只生成了V, 2.结构规范化去掉了E/T triple, 3.E/T link结果为空因而triple被删
    if not (has_entity or has_type):
        return []
    # 通过product枚举iter_list, 展开出多种sparql_struct候选
    return list(product(*iter_list))


def generate_neqg_link_results(params):
    # 实现NE-QG-EL流程, 将结果文件输出至../result/NEQG_inference_results/中
    # 参考eval_neqg运行模型, 规范化sparql_struct
    # 利用lookup_entity_linker实现entity&type的链接

    #idx2global_id = get_idx2global_id(raw_fn='../data/NEQG_gold/Test_QG_gold.json')
    info_data, dev_idx2global_id, test_idx2global_id = get_raw_info()
    if params['dev_or_test'] == 'dev':
        print('Evaluating dev set.')
        idx2global_id = dev_idx2global_id
    else:
        print('Evaluating test set.')
        idx2global_id = test_idx2global_id
    params = load_models_and_tokenizers(params)
    generated_neqg_link_results = []

    # 运行模型得到prediction (NE, QG, golds)
    get_device = lambda model : next(model.parameters()).device if model is not None else None
    all_sents, pred_queries, pred_ptrs, gold_queries, gold_ranges, pred_nodes, valid_len, pred_scores =\
        run_model_local(
            params['qg_model'], params['{}_loader'.format(params['dev_or_test'])], get_device(params['qg_model']),\
            ne_model=params['ne_model'], ne_device=get_device(params['ne_model']),\
            use_beam_search=params['use_beam_search'], num_beams=params['num_beams'], top_k=params['beam_search_top_k']
        )

    # 仿照QG.eval, 进行结构规范化, 输出Sparql_struct
    tokenizer = params['tokenizer']
    with open('../data/meta/test_meta.json') as fin:
        test_meta = json.load(fin)
    assert(len(all_sents) == len(pred_queries) == len(pred_ptrs) == len(gold_queries) == len(gold_ranges) == len(pred_nodes) == len(valid_len) == len(pred_scores))
    for cur_idx, sent, queries, pointers, gold_query, gold_range, node_label, cur_len, cur_scores\
        in tqdm(zip(range(len(all_sents)), all_sents, pred_queries, pred_ptrs, gold_queries, gold_ranges, pred_nodes, valid_len, pred_scores)):
        
        # 准备工作: 截断padding, 恢复问题, 类型判别 
        cur_labels = node_label[1:cur_len+1]
        cur_ques = tokenizer.convert_ids_to_tokens(sent[1:cur_len+1])        
        cur_nodes = get_nodes(cur_labels, cur_ques)
        # adjust cur_nodes to make it compatible with generated pointers and gold_ranges!
        for i in range(len(cur_nodes)):
            cur_nodes[i][1][0] += 1     # 因为pointer是包含<s>的, 亦即第一个token的ix是1
            cur_nodes[i][1][1] += 1     # 原来+=2, 是因为raw_data就是开区间; 现在raw是闭区间, 所以+=1即可
        
        raw_ques = tokenizer.decode(sent[1:cur_len+1])
        is_ask, is_count = judge_ask(raw_ques), judge_count(raw_ques)
        # 向params填充EL及expand所需的信息
        params['question'], params['is_ask'] = raw_ques, is_ask
        params['sent'], params['valid_len'] = sent, cur_len
        #if is_ask != test_meta[cur_idx]['is_ask']:
        #    print('ask false', test_meta[cur_idx])
        #if is_count != test_meta[cur_idx]['is_count']:
        #    print('count false', test_meta[cur_idx])

        # 与eval_neqg相同, 生成结点序列
        sparql_structs = []     # 输出sparql_struct作为NEQGEL结果
        sparql_scores = []      # 每个struct对应score, 由bs+el共同决定, 但目前没有考虑el的得分
        # 已经加入了的sparql_struct, 用于去重
        # 由于link_and_expand_triples中可能去除某些无链接结果triple, 这里记录所有expanded_structs[0]以去重; 注意此时两个expanded_structs相同等价于两个expanded_structs[0]相同
        added_structs = []
        for struct_idx, (query, pointer, bs_score) in enumerate(zip(queries, pointers, cur_scores)):

            # Step 2.1: match each <type, ptr> to a specific node in cur_nodes
            matched_nodes = match_ptr_to_node(pointer, query, cur_nodes)
            # Step 2.2: 与Linking同步, 进行结构规范化, 处理自环等错误/无意义情况
            # 注意到这里matched_nodes与原始QG输出可能不一样了, 是NEQG的最终结果!
            try:
                matched_nodes = struct_regularization(matched_nodes, is_ask, check_ask_struct=struct_idx==0)
            except ValueError:
                pass
                print("Ix = {}".format(idx2global_id[cur_idx]))

            # 将gold数据格式转换为与matched_nodes相同的 [[tag, [l, r]], ...]
            gold_nodes = [[tag, [l_pos, r_pos]] for tag, l_pos, r_pos in zip(gold_query, gold_range[0], gold_range[1])]
            gold_nodes = sort_node_seq_by_ptr(gold_nodes)

            # 为确保对齐, 这里再统一对node_sequence排序一下; 原因见sort_node_seq_by_ptr函数
            pre_matched_nodes = copy.deepcopy(matched_nodes)
            matched_nodes = sort_node_seq_by_ptr(matched_nodes)
            assert(matched_nodes == pre_matched_nodes)  # 在struct_regularization里其实已经sort了, 这里double check一下

            # link所有E/V并排列组合, 生成sparql_struct(s)
            cur_struct = []
            triples = nodes_to_triples(matched_nodes)
            expanded_structs = link_and_expand_triples(triples, params)
            if len(expanded_structs) == 0 or expanded_structs[0] in added_structs:
                continue
            added_structs.append(expanded_structs[0])
            sparql_structs += expanded_structs
            # TODO: 这里没有考虑link score, 且组合顺序为由前到后字典序, 需要考虑一下
            # 目前, 所有link结果的score都认为相同, 仅由bs_score决定sparql得分
            sparql_scores += [bs_score] * len(expanded_structs)

            if params['top1_struct']:
                break

        # 输出Sparql_stuct
        cur_result = dict()
        cur_result['_id'] = idx2global_id[cur_idx]
        cur_result['Question'] = raw_ques
        cur_result['Original sparql'] = info_data[cur_result['_id']]['Original sparql']
        cur_result['Readable_sparql'] = info_data[cur_result['_id']]['Readable_sparql']
        cur_result['is_ask'] = is_ask
        cur_result['is_count'] = is_count
        cur_result['Sparql_struct'] = sparql_structs
        cur_result['Sparql_scores'] = sparql_scores
        #cur_result['queries'] = queries
        #cur_result['pointers'] = pointers
        #cur_result['nodes'] = cur_nodes
        generated_neqg_link_results.append(cur_result)
    
    # 输出NEQGEL结果
    with open(params['output_fn'], 'w') as fout:
        json.dump(generated_neqg_link_results, fout, indent=4)


def main():
    params = {
        'model_config_fn': '../data/checkpoint/QG_Roberta_dev/config.json',
        'output_fn': '../result/NEQG_inference_results/{}_neqg_link_1219_dec.json',
        'use_beam_search': True,
        'num_beams': 4,
        'beam_search_top_k': 4,
        'label_topk': 20,
        'query_topk': 20,
        'add_format_results': True,
        'name_align': True,
        'el_topk': 2,
        'ask_full_cand': False,
        'top1_struct': True,
        'dev_or_test': 'dev',
    }
    params['output_fn'] = params['output_fn'].format(params['dev_or_test'])
    generate_neqg_link_results(params)

    # Update cache
    #params['linker'].flush_cache()
    #params['kb_client'].update_cache()


if __name__ == '__main__':
    main()
