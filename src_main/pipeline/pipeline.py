import sys
sys.path.insert(0, '..')
import json
import copy
from tqdm import tqdm

from pipeline.tf_ne_qg_link import get_raw_info
from RE_Cand.inference import QueryGraph, sparql_struct_preprocess, get_pkubase_client, get_kbqa_client


def get_sparql_answer_set(gc, sparql, key, kb='dbpedia_2016_04_core'):
    res_dict = json.loads(gc.query_with_cache(kb, "json", sparql))
    if res_dict['StatusCode'] != 0:
        print('Sparql status failed:\n{}'.format(sparql))
        return set()
    res_set = set()
    for item in res_dict['results']['bindings']:
        if key not in item:
            print(key, sparql, res_dict['results']['bindings'])
            assert(False)
        res_set.add(item[key]['value'])
    return res_set


def get_gold_answers(all_data, gc):
    # assume gold_sparql = all_data[ix]['Original sparql']
    all_answers = []
    print('Generating gold answers...')
    for entry in tqdm(all_data):
        gold_is_count = False
        gold_sparql = entry['Original sparql']
        if 'COUNT(?uri)' in gold_sparql:
            gold_sparql = gold_sparql.replace('COUNT(?uri)', '?uri')
            gold_is_count = True
        gold_target = 'askResult' if ('ASK WHERE' in gold_sparql) else 'uri'
        gold_answer_set = get_sparql_answer_set(gc, gold_sparql, gold_target)
        # count答案只包括一个int
        if gold_is_count:
            all_answers.append({len(gold_answer_set)})
        else:
            all_answers.append(gold_answer_set)
        if len(gold_answer_set) == 0:
            print('No answer for {}'.format(entry['_id']))
        if ('ASK WHERE' in gold_sparql):
            assert(gold_answer_set == {True})
    return all_answers


def cal_prf(answer_set, gold_answers):
    if len(answer_set) == len(gold_answers) == 0:
        p, r = 1., 1.
    elif len(answer_set) == 0:
        p, r = 0., 0.
    elif len(gold_answers) == 0:
        p, r = 0., 0.
    else:
        overlap_answers = answer_set.intersection(gold_answers)
        p = len(overlap_answers) / len(answer_set)
        r = len(overlap_answers) / len(gold_answers)
    f = 0 if p == r == 0 else 2 * p * r / (p + r)
    return p, r, f


def back_shift_type_triples(sparql_struct):
    # 1. 将包含type节点的triple移到最后
    # 2. 若当前sparql_struct为多跳且只有一个type, 其他都是variable, 则直接返回None跳过此候选(因为此候选会超时, 且只能得到p~0/r=1的结果, 意义不大)
    type_triples, other_triples, is_type_only = [], [], True
    for cur_triple in sparql_struct[1:]:
        if cur_triple[0][1] == 'type' or cur_triple[1][1] == 'type':
            type_triples.append(cur_triple)
        else:
            if cur_triple[0][1] != 'variable' or cur_triple[1][1] != 'variable':
                is_type_only = False
            other_triples.append(cur_triple)
    if len(sparql_struct) > 2 and len(type_triples) == 1 and is_type_only:
        return None
    return [sparql_struct[0]] + other_triples + type_triples


def query_re_structs(candidate_triples, cur_sparql_prefix, target, gc, is_ask):
    # RE模块将sparql_struct展开成有relation的多个triples, 此函数逐一查询, 若得到有答案的则返回
    answer_set, cur_sparql = set(), 'ask where {}'
    for sparql_struct_re in candidate_triples:
        cur_sparql = copy.deepcopy(cur_sparql_prefix)
        for cur_triple in sparql_struct_re:
            cur_sparql += ' '.join(cur_triple) + '. '
        cur_sparql += '}'

        # query KB, early stopping if has answer
        res_set = get_sparql_answer_set(gc, cur_sparql, target)
        if is_ask:
            res_list = list(res_set)
            assert(len(res_list) == 1 and type(res_list[0]) == bool)
            if res_list[0]:     # stop if find a True match
                answer_set.add(True)
                break
        elif len(res_set) > 0:
            answer_set = res_set
            break
    return answer_set, cur_sparql


def gen_re_result_early_stop(params):
    # 对于靠前的sparql_struct若已经有答案则提前结束, 主要对ASK类问题剪枝
    with open(params['neqglk_fn']) as fin:
        generated_neqg_link_results = json.load(fin)

    # setup connectors to servers
    gc = get_pkubase_client(
        host = params['kb_host_ip'],
        port = params['kb_host_port'])
    client = get_kbqa_client(
        host = params['re_host_ip'],
        port = params['re_host_port'])
    # get all gold answers
    gold_answers_list = get_gold_answers(generated_neqg_link_results, gc)
    #gc.update_cache()
    #exit(0)
    p, r, f = 0, 0, 0
    skip_ids = []
    skip_ids.append('591')
    #skip_ids.append('2395')
    for idx, (entry, gold_answers) in (enumerate(zip(generated_neqg_link_results, gold_answers_list))):
        #if entry['_id'] != '4723':
        #    continue
        question = entry["Question"]
        candidate_structs = entry["Sparql_struct"]
        entry['Candidate triples'] = []
        answer_set = set()
        cur_sparql = 'ask where {}'
        # 对prun的处理: 用一个list收集prun后的sparql, 后移运行
        prunned_infos = []
        for sparql_struct in candidate_structs:
            if entry['_id'] in skip_ids:
                break
            # 先运行完所有非prun的structs, 然后再跑prun的, 所以此处需同时记录其target_node和re_structs
            prunned_info = {}
            # 解决超时: 将type类triple向后靠; 若多跳且只有一个type, 直接跳过这个sparql
            sparql_struct = back_shift_type_triples(sparql_struct)
            if sparql_struct is None:
                continue
            # preprocess: add <> to all entity/type!
            sparql_struct = sparql_struct_preprocess(sparql_struct)

            # RE module to generate candidates (via beam search)
            qg = QueryGraph(sparql_struct, question, params['re_topk'])
            qg.generate_re_result(gc, client, pruning=True)            
            candidate_triples = copy.deepcopy(qg.beams)
            candidate_scores = copy.deepcopy(qg.scores)
            assert(len(candidate_triples) == len(candidate_scores))
            entry['Candidate triples'].append(candidate_triples)

            # construct sparqls
            if entry['is_ask']:
                cur_sparql_prefix = 'ask where { '
                target = 'askResult'
            else:
                cur_sparql_prefix = 'select distinct {} where {{ '.format(sparql_struct[0][0])
                target = sparql_struct[0][0][1:]    # remove '?'
            
            # 对prun的处理: 若RE后的sparql长度与link后的sparql_struct不一样, 后移
            # RE模块的性质: prun以neqglk_struct为单位, 产生的所有structs长度相同
            if len(candidate_triples[0]) + 1 != len(sparql_struct):
                for sparql_struct_re in candidate_triples:
                    assert(len(sparql_struct_re) + 1 != len(sparql_struct))
                prunned_info['cur_sparql_prefix'] = cur_sparql_prefix
                prunned_info['target'] = target
                prunned_info['candidate_triples'] = candidate_triples
                prunned_infos.append(prunned_info)
                continue
            
            answer_set, cur_sparql = query_re_structs(candidate_triples, cur_sparql_prefix, target, gc, entry['is_ask'])
            # if already get a valid answer, perform early stopping
            if len(answer_set):
                entry['Sparql'] = cur_sparql
                break
        
        # 若非prun的没有答案, 则继续查prun后的structs
        if len(answer_set) == 0:
            for prunned_info in prunned_infos:
                cur_sparql_prefix = prunned_info['cur_sparql_prefix']
                target = prunned_info['target']
                candidate_triples = prunned_info['candidate_triples']
                answer_set, cur_sparql = query_re_structs(candidate_triples, cur_sparql_prefix, target, gc, entry['is_ask'])
                # if already get a valid answer, perform early stopping
                if len(answer_set):
                    entry['Sparql'] = cur_sparql
                    break
        # dummy FALSE sparql
        if len(answer_set) == 0:
            entry['Sparql'] = cur_sparql
            if entry['is_ask']:
                #assert(False)
                answer_set.add(False)
        
        # cal P/R/F
        if entry['is_count']:
            answer_set = {len(answer_set)}
        cur_p, cur_r, cur_f = cal_prf(answer_set, gold_answers)
        p, r, f = p + cur_p, r + cur_r, f + cur_f
        entry['score'] = [round(cur_p, 4), round(cur_r, 4), round(cur_f, 4)]
        print(idx, cur_p, cur_r, cur_f)
        print('up to now: P/R/F = {}/{}/{}'.format(p/(idx+1), r/(idx+1), f/(idx+1)))

    p, r, f = p / len(generated_neqg_link_results), r / len(generated_neqg_link_results), f / len(generated_neqg_link_results)
    print('P/R/F = {}/{}/{}'.format(p, r, f))
    
    #gc.update_cache()
    with open(params['output_fn'], 'w') as fout:
        json.dump(generated_neqg_link_results, fout, indent=4)


def main():
    params = {
        'neqglk_fn': '../result/NEQG_inference_results/test_neqg_link_1219_cmtl.json',
        'output_fn': '../result/RE_inference_results/test_re_1219_cmtl.json',
        're_topk': 4,
        'kb_host_ip': '115.27.161.37',
        'kb_host_port': 9275,
        're_host_ip': 'localhost',
        're_host_port': 9305,
    }
    gen_re_result_early_stop(params)


if __name__ == '__main__':
    main()
