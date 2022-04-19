# Evaluation routines for tf-based QG.
# NOTE that the evaluation matrices can be directly compared with raw QG's eval_neqg
# Since the score matrix given by QG_TF model will be decoded to a node sequence assisted by NE node list

import copy
from collections import defaultdict
import torch

from NE.eval import get_nodes
from QG_table_filling.util import qg_tag2id, qg_id2tag   # For raw evaluation
from Sent_CLS.classify_question import judge_ask


# Translate NE tag names to qg_tag_id
ne_tag_to_qg_tag = {
    'E': 'entity',
    'V': 'variable',
    'VT': 'variable',
    'CT': 'type',
}

# NE results cache
ne_preds_cache = []


def run_model_local(model, dev_loader, ne_model=None, use_ne_cache=False):
    # 在本地加载评估集loader并运行模型，返回模型的NE、QG_TF预测结果

    get_device = lambda model : next(model.parameters()).device if model is not None else None
    device, ne_device = get_device(model), get_device(ne_model)
    model.eval()
    if ne_model:
        ne_model.eval()
    all_sents, score_matrices = [], []
    gold_queries, gold_ranges, valid_len = [], [], []
    gold_ne_tags = []
    global ne_preds_cache
    if ne_preds_cache == []:    # Initiate cache if empty
        print('Init NE cache')
        use_ne_cache = False
    if not use_ne_cache:
        ne_preds_cache = []
    print('Use NE cache or not:', use_ne_cache)

    print("Start running model.\n")
    for ix, item in enumerate(dev_loader):
        print("\rRunning model... {}/{}".format(ix, len(dev_loader)), end='')
        
        if len(item) == 7:
            sents, masks, adj_matrices, matrix_masks, node_types, node_ptrs, node_ends = item
        else:
            sents, masks, ne_tags, adj_matrices, matrix_masks, node_types, node_ptrs, node_ends = item
        sents, masks = sents.to(device), masks.to(device)
        valid_len += [int(a.sum().item()-2) for a in masks]

        with torch.no_grad():
            # QG_TF model
            sm_biaff_scores, ne_logits = model(sents, masks)
            
            if not use_ne_cache:
                # Separate NE model
                if type(ne_model) != type(None):
                    _, ne_logits = ne_model(
                        enc_input_ids = sents.to(ne_device),
                        enc_attention_mask = masks.to(ne_device)
                    )
                # Save NE tags in cache
                ne_preds_cache += torch.max(ne_logits, 2).indices.tolist()
            if len(item) == 8:
                gold_ne_tags += ne_tags.tolist()

            # turn gold data to list and get rid of paddings/start_tag/end_tag
            for query, start, end in zip(node_types, node_ptrs, node_ends):
                gold_queries.append(query[1:len(end)+1].tolist())
                gold_ranges.append([start[1:len(end)+1].tolist(), end])
            
            # save biaff_score results
            score_matrices += sm_biaff_scores.tolist()

            # all input raw sentences ([CLS] and [SEP] included, since ptr offset counts them)
            all_sents += sents.tolist()

    pred_nodes = copy.deepcopy(ne_preds_cache)
    assert(len(all_sents)==len(score_matrices)==len(gold_queries)==len(gold_ranges)==len(pred_nodes)==len(valid_len))
    return all_sents, score_matrices, gold_queries, gold_ranges, pred_nodes, valid_len, gold_ne_tags


def sort_node_seq_by_ptr(matched_nodes):
    cur_triples = []
    # triple 内排序
    for ix in range(1, len(matched_nodes), 2):
        cur_triples.append(sorted([matched_nodes[ix], matched_nodes[ix+1]],\
            key=lambda x:[x[1][0], x[0]]))
    # triple 排序
    cur_triples.sort(key=lambda x:[x[0][1][0], x[1][1][0]])
    sorted_nodes = sum(cur_triples, [matched_nodes[0]])
    return sorted_nodes


def sort_node_seq_by_tag(matched_nodes):
    cur_triples = []
    # triple 内排序
    for ix in range(1, len(matched_nodes), 2):
        cur_triples.append(sorted([matched_nodes[ix], matched_nodes[ix+1]],\
            key=lambda x:[qg_id2tag[x[0]], x[1][0]]))
    # triple 排序
    cur_triples.sort(key=lambda x:[qg_id2tag[x[0][0]], qg_id2tag[x[1][0]], x[0][1][0], x[1][1][0]])
    return cur_triples


def judge_total_match(matched_nodes, gold_nodes):
    # 判断两个node序列是否完全一致, 用以计算total_acc
    return matched_nodes == gold_nodes


def judge_actual_match(matched_nodes, gold_nodes):
    # 判断两个node序列是否语义相等, 即除variable外tag&span均一致, 且variable结构一致
    # 用以计算actual_acc
    # 早期训练可能存在NE提供空node_set情况, 此时matched_nodes=[None]
    if matched_nodes[0] is None:
        return False

    # 先针对matched_nodes按照tag排序, 以计算var2id
    pred_var2id = {str(matched_nodes[0]):0}  # target node是第0个variable
    pred_triples = sort_node_seq_by_tag(matched_nodes)
    
    # 计算var2id, 并利用var2id将所有variable的span替换为[id, id], 生成用于计算actual_acc的node序列, pred_aligned_nodes
    pred_aligned_nodes = [[matched_nodes[0][0], 0, 0],]    # target_node 的 id==0!
    for head, tail in pred_triples:
        if qg_id2tag[head[0]] == 'variable':
            if str(head) not in pred_var2id.keys():
                pred_var2id[str(head)] = len(pred_var2id)
                assert(pred_var2id[str(head)] == len(pred_var2id)-1)
            aligned_head = [head[0], [pred_var2id[str(head)], pred_var2id[str(head)]]]
        else:
            aligned_head = copy.deepcopy(head)
        if qg_id2tag[tail[0]] == 'variable':
            if str(tail) not in pred_var2id.keys():
                pred_var2id[str(tail)] = len(pred_var2id)
                assert(pred_var2id[str(tail)] == len(pred_var2id)-1)
            aligned_tail = [tail[0], [pred_var2id[str(tail)], pred_var2id[str(tail)]]]
        else:
            aligned_tail = copy.deepcopy(tail)
        pred_aligned_nodes += [aligned_head, aligned_tail]
    # 替换完var_id, 其实只要按照相同的规则进行triple内+间排序, 相同的aligned_nodes即代表actual_match
    pred_aligned_nodes = sort_node_seq_by_ptr(pred_aligned_nodes)

    # 同理针对gold_nodes处理
    gold_var2id = {str(gold_nodes[0]):0}  # target node是第0个variable
    gold_triples = sort_node_seq_by_tag(gold_nodes)
    
    gold_aligned_nodes = [[gold_nodes[0][0], 0, 0],]    # target_node 的 id==0!
    for head, tail in gold_triples:
        if qg_id2tag[head[0]] == 'variable':
            if str(head) not in gold_var2id.keys():
                gold_var2id[str(head)] = len(gold_var2id)
                assert(gold_var2id[str(head)] == len(gold_var2id)-1)
            aligned_head = [head[0], [gold_var2id[str(head)], gold_var2id[str(head)]]]
        else:
            aligned_head = copy.deepcopy(head)
        if qg_id2tag[tail[0]] == 'variable':
            if str(tail) not in gold_var2id.keys():
                gold_var2id[str(tail)] = len(gold_var2id)
                assert(gold_var2id[str(tail)] == len(gold_var2id)-1)
            aligned_tail = [tail[0], [gold_var2id[str(tail)], gold_var2id[str(tail)]]]
        else:
            aligned_tail = copy.deepcopy(tail)
        gold_aligned_nodes += [aligned_head, aligned_tail]
    gold_aligned_nodes = sort_node_seq_by_ptr(gold_aligned_nodes)
    
    return pred_aligned_nodes == gold_aligned_nodes


def normalize_score_matrix(score_matrix, cur_len):
    # Remove paddings in score_matrix.
    # Note that <s> & </s> should be reserved for target & NONE node
    norm_score_matrix = [cur_row[:cur_len+2] for cur_row in score_matrix[:cur_len+2]]
    return norm_score_matrix


def normalize_ne_nodes(cur_nodes):
    # Normalize offsets of NE results [ ['E', [0, 1]], ... ]
    for i in range(len(cur_nodes)):
        cur_nodes[i][1][0] += 1     # include <s> offset
        cur_nodes[i][1][1] += 1     # closed range
    return cur_nodes


def cal_vote_prob(score_matrix, src_node, dst_node):
    # Average scores on square <sl,sr,dl,dr>, all ranges are CLOSED
    src_l, src_r = src_node[1][0], src_node[1][1]
    dst_l, dst_r = dst_node[1][0], dst_node[1][1]
    avg_score = 0.
    for ix in range(src_l, src_r + 1):
        for jx in range(dst_l, dst_r + 1):
            avg_score += score_matrix[ix][jx][1]
    avg_score /= (src_r - src_l + 1) * (dst_r - dst_l + 1)
    return avg_score


def cal_symm_vote_prob(score_matrix, node1, node2):
    # Cal vote prob of n1->n2 & n2->n1, return their avg
    score_1 = cal_vote_prob(score_matrix, node1, node2)
    score_2 = cal_vote_prob(score_matrix, node2, node1)
    return (score_1 + score_2) / 2


def select_target_node(score_matrix, cur_nodes, is_ask_query=False):
    # Decide select-variable, i.e. the node linked to <s> token
    # There should be ONE and ONLY ONE target, so select the nodes with highest avg prob
    # For ASK queries, target node is always </s>
    if is_ask_query:
        none_node = [qg_tag2id['variable'], [0, 0]]
        return none_node
    start_node = ['_', [0, 0]]
    max_vote_prob, target_node = -1, None
    for node in cur_nodes:
        cur_vote_prob = cal_symm_vote_prob(score_matrix, start_node, node)
        if cur_vote_prob > max_vote_prob:
            max_vote_prob = cur_vote_prob
            target_node = copy.deepcopy(node)
            target_node[0] = qg_tag2id['variable']
    return target_node


def build_edges(score_matrix, cur_nodes, thresh=0.5):
    # Returns the edge list between cur_nodes given score_matrix
    virtual_none_node = ['V', [len(score_matrix)-1, len(score_matrix)-1]]
    cur_nodes = cur_nodes + [virtual_none_node]
    edge_list = []
    for nix_i in range(len(cur_nodes)):
        node1 = cur_nodes[nix_i]
        for nix_j in range(nix_i, len(cur_nodes)):
            node2 = cur_nodes[nix_j]
            vote_prob = cal_symm_vote_prob(score_matrix, node1, node2)
            if vote_prob > thresh:
                edge_list.append([nix_i, nix_j])
    return edge_list


def struct_regularization(cur_nodes, edge_list, is_ask_query=False, remove_homo_vars=True):
    # Regularize decoded edge_list from score_matrix, rules resemble original QG
    # Outputs the corresponding node sequence
    # cur_nodes := [[node_type_id, [ptr_l, ptr_r (closed)]], ...]
    # edge_list := [[nid1, nid2], ...]
    # is_ask_query: if so, don't apply rule-3 & check if only ONE edge is generated
    # Rule-1:   delete duplicate triples (naturally accomplished)
    # Rule-2:   for self loops, only reserver VTs (if NE classify it as V, forcibly convert to VT)
    # Rule-3:   delete triples that don't contain variable
    # Rule-4:   remove homogeneous variables (TODO)
    # TODO check 1-hop an raise exceptions for ASK queries

    # The span of None-node should be finally [0,0], so do corrections here
    virtual_none_node = ['V', [0, 0]]
    cur_nodes = cur_nodes + [virtual_none_node]

    reg_edge_list, output_edge_list = [], []
    var_neighbors, vars_to_remove = defaultdict(set), []  # for judging homogeneous variables
    for nid1, nid2 in edge_list:
        # Rule-2
        if nid1 == nid2 and 'V' in cur_nodes[nid1][0]:  # V or VT
            reg_edge_list.append([nid1, nid2])
        # Rule-3
        if nid1 != nid2:
            node1, node2 = cur_nodes[nid1], cur_nodes[nid2]
            if 'V' not in node1[0] and 'V' not in node2[0] and (not is_ask_query):
                continue
            reg_edge_list.append([nid1, nid2])
            if 'V' in node1[0]:
                var_neighbors[nid1].add(nid2)
            if 'V' in node2[0]:
                var_neighbors[nid2].add(nid1)
            
    # Rule-4
    if remove_homo_vars:
        var_neighbors = list(var_neighbors.items())
        for iix, var1 in enumerate(var_neighbors):
            for jix in range(iix):
                var2 = var_neighbors[jix]
                # If var1 & var2 are homogeneous, i.e. their adjacent node sets are the same
                if var1[1] == var2[1]:
                    vars_to_remove.append(var2[0])
                    #assert(False)
        assert(set(list(set(vars_to_remove))) == set(vars_to_remove))

    for nid1, nid2 in reg_edge_list:
        # Remove a var node by removing all its adj edges
        if nid1 in vars_to_remove or nid2 in vars_to_remove:
            continue
        if nid1 == nid2:
            node1, node2 = copy.deepcopy(cur_nodes[nid1]), copy.deepcopy(cur_nodes[nid1])
            node1[0], node2[0] = qg_tag2id['variable'], qg_tag2id['type']
            output_edge_list.append([node1, node2])
        else:
            node1, node2 = copy.deepcopy(cur_nodes[nid1]), copy.deepcopy(cur_nodes[nid2])
            node1[0] = qg_tag2id[ne_tag_to_qg_tag[node1[0]]]
            node2[0] = qg_tag2id[ne_tag_to_qg_tag[node2[0]]]
            output_edge_list.append([node1, node2])
    
    return output_edge_list


def decode_score_matrix_with_ne_preds(score_matrix, cur_nodes, is_ask_query=False, remove_homo_vars=True):
    # Given NE nodes and score_matrix of QG_TF, generate all edges
    target_node = select_target_node(score_matrix, cur_nodes, is_ask_query=is_ask_query)
    edge_list = build_edges(score_matrix, cur_nodes)
    reg_edge_list = struct_regularization(cur_nodes, edge_list, is_ask_query=is_ask_query, remove_homo_vars=remove_homo_vars)
    return sum(reg_edge_list, [target_node])


def eval_neqg_tf(model, dev_loader, tokenizer, ne_model=None, output_full_result=False, remove_homo_vars=True, use_ne_cache=False):
    # evaluation function for the combined results of NE & Table-filling-QG
    
    # Get model predictions
    all_sents, score_matrices, gold_queries, gold_ranges, pred_node_labels, valid_len, gold_ne_tags = run_model_local(model, dev_loader, ne_model=ne_model, use_ne_cache=use_ne_cache)
    #return all_sents, score_matrices, gold_queries, gold_ranges, pred_node_labels, valid_len, gold_ne_tags

    total_acc, actual_acc = 0., 0.
    full_results = []
    for cur_idx, sent, score_matrix, gold_query, gold_range, node_label, cur_len in zip(range(len(all_sents)), all_sents, score_matrices, gold_queries, gold_ranges, pred_node_labels, valid_len):
        
        # Step 1: get all nodes using node_labels
        cur_labels = node_label[1:cur_len+1]
        cur_ques = tokenizer.convert_ids_to_tokens(sent[1:cur_len+1])
        raw_ques = tokenizer.decode(sent[1:cur_len+1])
        cur_nodes = get_nodes(cur_labels, cur_ques)
        cur_nodes = normalize_ne_nodes(cur_nodes)
        is_ask = judge_ask(raw_ques)

        # Step 2: decode score_matrix using NE nodes
        norm_score_matrix = normalize_score_matrix(score_matrix, cur_len)
        pred_node_seq = decode_score_matrix_with_ne_preds(norm_score_matrix, cur_nodes, is_ask_query=is_ask, remove_homo_vars=remove_homo_vars)
        
        # Acc evaluation
        # Convert gold data to node_seq, note that <start> is removed from gold_query in **run_model_local**
        gold_node_seq = [[tag, [l_pos, r_pos]] for tag, l_pos, r_pos in zip(gold_query, gold_range[0], gold_range[1])]

        # Sort to align pred & gold
        pred_node_seq = sort_node_seq_by_ptr(pred_node_seq)
        gold_node_seq = sort_node_seq_by_ptr(gold_node_seq)
        
        # Calculating accuracies
        total_match = judge_total_match(pred_node_seq, gold_node_seq)
        actual_match = judge_actual_match(pred_node_seq, gold_node_seq)
        total_acc += total_match
        actual_acc += actual_match
        if total_match:
            assert(actual_match)
        if output_full_result:
            full_results.append([sent, cur_len, raw_ques, pred_node_seq])
    
    # do average to get final score
    total_acc /= len(all_sents)
    actual_acc /= len(all_sents)
    
    model.train()
    if output_full_result == 'get_ne_cached_results':
        return total_acc, actual_acc, [pred_node_labels, gold_ne_tags, valid_len, all_sents]
    elif output_full_result:
        return total_acc, actual_acc, full_results
    else:
        return total_acc, actual_acc
