import sys
sys.path.append('../')
import json
from sklearn import metrics
import torch

from NE.util import id2tag


def show_scores(predictions, v_labels, valid_len, f):
    score_name = ['Micro precision', 'Macro precision', 'Micro recall', 'Macro recall',
              'Micro F1', 'Macro F1']
    scores = [0.]*6
    for preds, golds, v_len in zip(predictions, v_labels, valid_len):
        preds = preds[1:v_len+1]
        golds = golds[1:v_len+1]
        scores[0] += (metrics.precision_score(preds, golds, average='micro'))
        scores[1] += (metrics.precision_score(preds, golds, average='macro'))
        scores[2] += (metrics.recall_score(preds, golds, average='micro'))
        scores[3] += (metrics.recall_score(preds, golds, average='macro'))
        scores[4] += (metrics.f1_score(preds, golds, average='micro'))
        scores[5] += (metrics.f1_score(preds, golds, average='macro'))
    for i in range(len(scores)):
        scores[i] /= len(predictions)
    for na, sc in zip(score_name, scores):
        print(na, ': ', sc)
        f.write(na+': '+str(sc)[:7]+'\n')
    return scores


def get_nodes(labels, ques):
    # given a seqence of node-label IDs, extract all nodes in list form. Sample input:
    # lables = [1, 2, 2, 2, 2, 0, 3, 4, 4, 4, 0, 0, 0]
    # Sample return:
    # spans = [[0, 1], [8, 11]]
    # nodes = [['E', [0, 1]], ['V', [8, 11]]]
    # NOTE: assuming that NO PADDING is in labels & ques

    # preprocess before extracting all nodes
    state = 0
    for i, lab in enumerate(labels):
        if id2tag[lab] == 'O':
            state = 0
            continue
        if 'I' in id2tag[lab] and state==0:
            # the case when extra processing is needed
            # O->I
            state = 1
            if i>=1 and ("\"" in ques[i-1]):
                labels[i] = lab-1
                continue
            if i>=2 and id2tag[labels[i-2]] != 'O':
                labels[i-1] = lab
                continue
            # start with I
            labels[i] = lab-1
        state = 1
    
    # extract nodes, vote to get the type of each node
    if len(id2tag) == 3:
        node_types = ("_")
    else:
        node_types = ("E","V","VT","CT")
    nodes = []  # list of nodes, each item is a pair of <node_type, [begin, end]>
    counts = [0]*len(node_types)
    state = 0   # currently in a node or not
    pos = -1

    for i,lab in enumerate(labels+[0]):
        if id2tag[lab] == 'O':
            if state!=0:
                cur_type = node_types[counts.index(max(counts))]
                nodes.append([cur_type, [pos,i-1]])
                counts = [0]*len(node_types)
            state = 0
        elif 'B' in id2tag[lab]:
            if state!=0:
                cur_type = node_types[counts.index(max(counts))]
                nodes.append([cur_type, [pos,i-1]])
                counts = [0]*len(node_types)
            state = 1
            pos = i
            counts[(lab-1)//2] += 1
        elif id2tag[lab] not in ('<START>','<STOP>'):
            counts[(lab-1)//2] += 1

    return nodes


def node_evaluation(labels, ques, golds):
    # turn sequence-labling results to node extraction results, then evaluate.
    # handle invalid label subsequences using heuristic rules
    
    # get all nodes in list form, see **get_nodes** function!
    nodes = get_nodes(labels, ques)
    spans = [node[-1] for node in nodes]
    gold_nodes = get_nodes(golds, ques)
    gold_spans = [gold_node[-1] for gold_node in gold_nodes]

    #print(labels, golds, nodes, gold_nodes)
    cur_node_p, cur_node_r, cur_node_f = 0.,0.,0.
    cur_span_p, cur_span_r, cur_span_f = 0.,0.,0.
    for span, node in zip(spans, nodes):
        if node in gold_nodes:
            cur_node_p += 1
            cur_node_r += 1
        if span in gold_spans:
            cur_span_p += 1
            cur_span_r += 1
    
    if len(nodes) == 0:
        #print([id2tag[i] for i in labels],\
        #     [id2tag[i] for i in golds], ques, nodes, gold_nodes, spans, gold_spans)
        cur_node_p = 1.
    else:
        cur_node_p /= len(nodes)
    if len(gold_nodes) == 0:
        cur_node_r = 1.
        #assert(False)
    else:
        cur_node_r /= len(gold_nodes)

    if cur_node_p==0 or cur_node_r==0:
        cur_node_f = 0.
    else:
        cur_node_f = 2*cur_node_p*cur_node_r/(cur_node_p+cur_node_r)

    if len(spans)==0:
        cur_span_p = 1.
    else:
        cur_span_p /= len(spans)
    if len(gold_spans)==0:
        cur_span_r = 1.
        #assert(False)
    else:
        cur_span_r /= len(gold_spans)
    if cur_span_p==0 or cur_span_r==0:
        cur_span_f = 0.
    else:
        cur_span_f = 2*cur_span_p*cur_span_r/(cur_span_p+cur_span_r)

    return cur_node_p, cur_node_r, cur_node_f, cur_span_p, cur_span_r, cur_span_f


def run_model_local(model, dev_loader, device):
    # Run model & collect predictions
    model.eval()
    pred_tags, gold_tags, valid_len, all_sents = [], [], [], []

    # Model predication & other info
    with torch.no_grad():
        for cur_item in dev_loader:
            quess, masks, tags = cur_item
            quess, masks, tags = quess.to(device), masks.to(device), tags.to(device)
            # model output
            enc_output, ne_logits = model(quess, masks)
            
            # predictions w.r.t. max score
            pred_tags += ne_logits.max(dim=-1).indices.tolist()
            # correct tags on dev set
            gold_tags += tags.tolist()
            # since we must evaluate performance on raw sentence, get the length of raw sents (-2: <CLS> & <SEP>) 
            valid_len += [int(a.sum().item()-2) for a in masks]
            # all input raw sentences
            all_sents += quess.tolist()
    return pred_tags, gold_tags, valid_len, all_sents


def eval_ne(model, model_config, dev_loader, device, tokenizer, log_output_file,\
            output_model_file, epoch, best_matrics, save_res=False, cached_outputs=None):
    
    if model is not None:
        pred_tags, gold_tags, valid_len, all_sents = run_model_local(model, dev_loader, device)
    else:
        pred_tags, gold_tags, valid_len, all_sents = cached_outputs
    
    # Tag PRF
    f = open(log_output_file, 'a', encoding='utf-8')
    f.write("Epoch: "+str(epoch)+'\n')
    show_scores(pred_tags, gold_tags, valid_len, f)[-1]
    f.close()

    # Span P/R/F
    node_p, node_r, node_f, span_p, span_r, span_f = 0,0,0,0,0,0
    for cur_preds, cur_golds, cur_len, cur_sent in zip(pred_tags, gold_tags, valid_len, all_sents):
        cur_labels = cur_preds[1:cur_len+1]
        cur_golds = cur_golds[1:cur_len+1]
        cur_ques = tokenizer.convert_ids_to_tokens(cur_sent[1:cur_len+1])
        p1,r1,f1,p2,r2,f2 = node_evaluation(cur_labels, cur_ques, cur_golds)
        node_p += p1
        node_r += r1
        node_f += f1
        span_p += p2
        span_r += r2
        span_f += f2
    
    node_p /= len(pred_tags)
    node_r /= len(pred_tags)
    node_f /= len(pred_tags)
    span_p /= len(pred_tags)
    span_r /= len(pred_tags)
    span_f /= len(pred_tags)

    f = open(log_output_file, 'a', encoding='utf-8')
    print("Node P: %s, R: %s, F: %s."%(str(node_p)[:8],str(node_r)[:8],str(node_f)[:8]))
    print("Span P: %s, R: %s, F: %s."%(str(span_p)[:8],str(span_r)[:8],str(span_f)[:8]))
    f.write("Node P: %s, R: %s, F: %s.\n"%(str(node_p)[:8],str(node_r)[:8],str(node_f)[:8]))
    f.write("Span P: %s, R: %s, F: %s.\n"%(str(span_p)[:8],str(span_r)[:8],str(span_f)[:8]))
    
    if model_config.get('original_ne', True):
        print("Current best Node Recall: " + str(max(node_r, best_matrics))[:9])
        f.write("Best Node Recall: " + str(max(node_r, best_matrics))[:9]+'\n')
        cur_metrics = node_r
    else:
        print("Current best Node F1: " + str(max(node_f, best_matrics))[:9])
        f.write("Best Node F1: " + str(max(node_f, best_matrics))[:9]+'\n')
        cur_metrics = node_f
    f.close()

    # Save model
    if cur_metrics > best_matrics:
        best_matrics = cur_metrics
        if save_res:
            print('Saving model...')
            torch.save(model.state_dict(), output_model_file)
    if model is not None:
        model.train()
    return best_matrics
