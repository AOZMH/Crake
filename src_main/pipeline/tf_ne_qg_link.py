# See **eval.py** in NE/QG_table_filling for more details
# Do NE+QG (and corresponding postprocessing) and entity linking in pipeline

import sys
sys.path.insert(0, '..')
import json
import copy
from itertools import product
from tqdm import tqdm
import torch

from QG_table_filling.eval import eval_neqg_tf
from QG_table_filling.util import qg_id2tag
from QG_TF_NE_multitask.main import build_dataloaders_and_models_qg
from Sent_CLS.classify_question import judge_ask, judge_count
from lookup_entity_linker import get_pkubase_client, dbpedia_lookup_entity_linker, link_entity, link_type


def load_models_and_tokenizers(model_config):
    # Load qg_tf & ne model checkpoint, dataloaders and entity linker
    
    # Dataloaders & models
    model_config = build_dataloaders_and_models_qg(model_config)
    # Device transfer
    main_gpu = [int(gpu_id) for gpu_id in model_config.get('gpus', '0,1').split(',')][0]
    ne_gpu = [int(gpu_id) for gpu_id in model_config.get('ne_gpus', '0,1').split(',')][-1]
    use_ne_cache = model_config.get('use_ne_cache', False)
    device = torch.device("cuda:{}".format(main_gpu) if torch.cuda.is_available() else "cpu")
    ne_device = torch.device("cuda:{}".format(ne_gpu) if torch.cuda.is_available() and (not use_ne_cache) else "cpu")
    print("QG model running on gpu:", device)
    model_config['model'] = model_config['model'].to(device)
    if model_config['ne_model']:
        print("NE model running on gpu:", ne_device)
        model_config['ne_model'] = model_config['ne_model'].to(ne_device)
    
    # Entity linking (with cache)
    model_config['linker'] = dbpedia_lookup_entity_linker(
        url = model_config['lookup_url'],
        cache_fn = model_config['lookup_cache_fn'])
    # KB client for entity linking (with cache)
    model_config['kb_client'] = get_pkubase_client(
        host = model_config['kb_host_ip'],
        port = model_config['kb_host_port'])
    # 链接需要用到train数据
    train_type_mapping_fn = '../data/mappings/train_type_mappings_ques.json'
    with open(train_type_mapping_fn) as fin:
        model_config['train_type_mappings'] = json.load(fin)
    
    return model_config


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


def generate_tf_neqg_link_results(model_config):
    # NE-(TF-based) QG-Linking
    # Results are saved to ../result/NEQG_inference_results/

    model_config = load_models_and_tokenizers(model_config)
    info_data, dev_idx2global_id, test_idx2global_id = get_raw_info()
    if model_config['dev_or_test'] == 'dev':
        print('Evaluating dev set.')
        idx2global_id = dev_idx2global_id
    else:
        print('Evaluating test set.')
        idx2global_id = test_idx2global_id

    # Directly get node_sequences
    if hasattr(model_config['model'], 'set_ne_label_encoder_mode'):
        model_config['model'].set_ne_label_encoder_mode('test')
    total_acc, actual_acc, pred_outputs = eval_neqg_tf(model_config['model'], model_config['{}_loader'.format(model_config['dev_or_test'])], model_config['tokenizer'], ne_model=model_config['ne_model'], remove_homo_vars=model_config.get('remove_homo_vars', False), output_full_result=True)
    print("Total Acc: {}, Actual Acc: {}.".format(total_acc, actual_acc))

    # Linking & output
    generated_neqg_link_results = []
    for cur_idx, pred_output in enumerate(tqdm(pred_outputs)):
        # Get predictions (question, decoded table -> pred_node_seq)
        sent, cur_len, raw_ques, pred_node_seq = pred_output
        is_ask, is_count = judge_ask(raw_ques), judge_count(raw_ques)

        # Update params for entity-linking & expand
        model_config['question'], model_config['is_ask'] = raw_ques, is_ask
        model_config['sent'], model_config['valid_len'] = sent, cur_len
        
        # link all E/V & product on link-candidates, generate sparql_struct(s)
        triples = nodes_to_triples(pred_node_seq)
        expanded_structs = link_and_expand_triples(triples, model_config)
        sparql_structs = expanded_structs   # Only 1 struct for TF-QG
        # TODO no link scores provided currently, treat as uniform
        sparql_scores = [1.] * len(expanded_structs)

        # Output Sparql_stuct
        cur_result = dict()
        cur_result['_id'] = idx2global_id[cur_idx]
        cur_result['Question'] = raw_ques
        cur_result['Original sparql'] = info_data[cur_result['_id']]['Original sparql']
        cur_result['Readable_sparql'] = info_data[cur_result['_id']]['Readable_sparql']
        cur_result['is_ask'] = is_ask
        cur_result['is_count'] = is_count
        cur_result['Sparql_struct'] = sparql_structs
        cur_result['Sparql_scores'] = sparql_scores
        generated_neqg_link_results.append(cur_result)
    
    # 输出NEQGEL结果
    with open(model_config['output_fn'], 'w') as fout:
        json.dump(generated_neqg_link_results, fout, indent=4)


def main():
    params = {
        'output_fn': '../result/NEQG_inference_results/{}_neqg_link_1219_cmtl.json',
        'label_topk': 20,
        'query_topk': 20,
        'add_format_results': True,
        'name_align': True,
        'el_topk': 2,
        'ask_full_cand': False,
        'dev_or_test': 'test',
        'kb_host_ip': '115.27.161.37',
        'kb_host_port': 9275,
        'lookup_url': 'http://115.27.161.37:9273/lookup-application/api/search',
        'lookup_cache_fn': '../data/lookup_cache/cache.dat',
    }
    params['output_fn'] = params['output_fn'].format(params['dev_or_test'])
    # Merge with model_config infos
    config_fn = '../data/checkpoint/QG_TF_NE_cmtl_fcn_no_pretrain/config.json'
    with open(config_fn) as fin:
        model_config = json.load(fin)
        model_config['do_eval_only'] = True
    for ck, cv in model_config.items():
        assert(ck not in params), (ck, cv)
        params[ck] = cv
    generate_tf_neqg_link_results(params)

    # Update cache
    #params['linker'].flush_cache()
    #params['kb_client'].update_cache()


if __name__ == '__main__':
    main()
