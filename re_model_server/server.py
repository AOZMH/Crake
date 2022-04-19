import sys
sys.path.append('/home/zhangminhao/.conda/envs/asdf/lib/python3.7/site-packages/')
import copy
import json
import socket
import socketserver
from threading import Lock
import time
import struct
import traceback
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers

import model
from helper import get_nodes, id2tag, tag2id, qg_tag2id,  MyDataset, custom_collate, load_all_relations, modify_socket_buffer_size, resolve_rel, recv_n_bytes

re_model, neqg_model = None, None


class KBQA_Models(object):
    def __init__(self, model_config_fn):
        with open(model_config_fn, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)
        self.model = None
        self.tokenizer = None
        gpus = self.model_config.get('gpus', '3')
        self.gpu_id = int(gpus.split(',')[-1])
        self.model_lock = Lock()
        self.re_cache_lock = Lock()
        self.re_cache_fn = self.model_config.get('re_cache_fn', None)
        if not self.re_cache_fn:
            self.re_cache = dict()
        else:
            try:
                with open(self.re_cache_fn, 'r', encoding='utf-8') as fin:
                    self.re_cache = json.load(fin)
                print('RE cache loaded from {}.'.format(self.re_cache_fn))
            except FileNotFoundError:
                self.re_cache = dict()
                print("Empty cache initiated")
        # Load default relations
        self.all_relations_fn = self.model_config.get('all_relations_fn', './data/all_relations.dat')
        self.all_relations = load_all_relations(self.all_relations_fn)
        self.is_uri_all_relations = self.model_config.get('is_uri_all_relations', False)
        if self.is_uri_all_relations:
            # Prepare string-format default relations & uri-str convertion dict
            self.all_relations_str, self.str_to_uri = [], {}
            for rel in self.all_relations:
                rel_info = resolve_rel(rel)
                assert(rel_info.get('success', True))    # Default relations must be resolvable when initializing
                rel_str = rel_info['rel_type'] + ' ' + rel_info['rel_name']
                self.all_relations_str.append(rel_str)
                self.str_to_uri[rel_str] = rel
        print("Model running on gpu {}.".format(self.gpu_id))
    
    def lazy_load_model(self):
        # lazy loading
        self.model_lock.acquire()
        if not self.tokenizer:
            tokenizer_class = getattr(transformers, self.model_config.get('tokenizer_class', 'BertTokenizer'))
            tokenizer_route = self.model_config["tokenizer_route"]
            self.tokenizer = tokenizer_class.from_pretrained(tokenizer_route)
        if not self.model:
            model_class = getattr(model, self.model_config.get('model_class', 'Multitask_Model'))
            self.model = model_class(self.model_config)
            output_model_file = self.model_config['checkpoint_route']
            self.model.load_state_dict(torch.load(output_model_file, map_location=torch.device('cpu')), strict=False)
            self.model = self.model.cuda(self.gpu_id)
            self.model.eval()
            # Using different NE & QG model
            if 'ne_model_config_route' in self.model_config.keys():
                with open(self.model_config['ne_model_config_route']) as fin:
                    ne_model_config = json.load(fin)
                self.ne_model = model_class(ne_model_config)
                print("Loading assisting NE model from {}...".format(ne_model_config['checkpoint_route']))
                self.ne_model.load_state_dict(torch.load(ne_model_config['checkpoint_route'], map_location=torch.device('cpu')))
                self.ne_model = self.ne_model.cuda(self.gpu_id)
                self.ne_model.eval()
        self.model_lock.release()
    
    def extract_node(self, sent, verbose=True):
        # Get a question in string form, return the nodes and the type of the nodes in it        
        self.lazy_load_model()

        # tokenize data and form input
        q_tok = self.tokenizer.encode(list(sent))
        q_input = torch.LongTensor([q_tok])
        q_att_mask = torch.FloatTensor([[1]*len(q_tok)])
        q_input, q_att_mask = q_input.cuda(self.gpu_id), q_att_mask.cuda(self.gpu_id)

        print("Start extracting nodes...")
        self.model_lock.acquire()
        with torch.no_grad():
            logits,_ = self.model(q_input, q_att_mask)
                
            # prediction with regard to max score
            pred = torch.max(logits, 2)[1].tolist()
        self.model_lock.release()
        
        if verbose:
            for tok in list(sent):
                print(tok,end=' ')
            print()
            for label in pred[0][1:-1]:
                print(id2tag[label]+' ',end='')
            print()

        labels = pred[0][1:-1]
        ques = list(sent)
        nodes = get_nodes(labels, ques)
        for ix in range(len(nodes)):
            nodes[ix].insert(0, ''.join(ques[nodes[ix][-1][0]:nodes[ix][-1][1]+1]))

        if verbose:
            print(nodes)
        return nodes

    def generate_query_and_nodes(self, sent, use_beam_search=True, num_beams=4, top_k=4):
        # Extract nodes and generate node_tag & pointer sequence of a given sentence
        self.lazy_load_model()

        # tokenize data and form input
        q_tok = self.tokenizer.encode(list(sent))
        q_input = torch.LongTensor([q_tok])
        q_att_mask = torch.FloatTensor([[1]*len(q_tok)])
        q_input, q_att_mask = q_input.cuda(self.gpu_id), q_att_mask.cuda(self.gpu_id)

        print("Start extracting nodes...")
        self.model_lock.acquire()
        with torch.no_grad():
            if use_beam_search:
                res_queries, res_pointers, ne_logits, topk_score = self.model.beam_search_generate(q_input, enc_attention_mask=q_att_mask,\
                dec_bos_id=qg_tag2id['<start>'], dec_eos_id=qg_tag2id['<end>'], sorted_seq=True, num_beams=num_beams, top_k=top_k)
                topk_score = topk_score.tolist()
            else:
                res_queries, res_pointers, ne_logits = model.generate(q_input, enc_attention_mask=q_att_mask,\
                dec_bos_id=qg_tag2id['<start>'], dec_eos_id=qg_tag2id['<end>'], sorted_seq=True)
                res_queries = [[q] for q in res_queries]
                res_pointers = [[p] for p in res_pointers]
                topk_score = [[1] for i in range(len(res_queries))]
            
            # If using a separate NE model
            if hasattr(self, 'ne_model'):
                ne_logits,_ = self.ne_model(q_input, enc_attention_mask=q_att_mask)
            
        self.model_lock.release()

        # format return values
        node_tags = torch.max(ne_logits, 2).indices.tolist()[0]
        pred_scores = topk_score[0]
        pred_queries = res_queries[0]
        pred_ptrs = res_pointers[0]
        
        # also add NE resutls for better visualization
        ques = list(sent)
        nodes = get_nodes(node_tags[1:-1], ques)
        for ix in range(len(nodes)):
            nodes[ix].insert(0, ''.join(ques[nodes[ix][-1][0]:nodes[ix][-1][1]+1]))
        
        return node_tags, pred_scores, pred_queries, pred_ptrs, nodes
        
    def extract_relation(self, ques, m1, m2, cand_rel, K):
        # Extract most probable relations from candidates
        self.lazy_load_model()
        
        cur_ques = self.tokenizer.encode(ques, add_special_tokens=False)
        cur_rels = [self.tokenizer.encode(rel, add_special_tokens=False) for rel in cand_rel]
        cur_m1 = self.tokenizer.encode(m1, add_special_tokens=False)
        cur_m2 = self.tokenizer.encode(m2, add_special_tokens=False)

        cur_sent_in = [self.tokenizer.cls_token_id]+cur_ques+[self.tokenizer.sep_token_id]+cur_m1+\
                [self.tokenizer.sep_token_id]+cur_m2+[self.tokenizer.sep_token_id]
        
        toks, type_masks = [],[]
        for cur_rel in cur_rels:
            cur_rel_in = cur_rel+[self.tokenizer.sep_token_id]
            # type mask channel
            cur_type_mask = [0]*len(cur_sent_in)+[1]*len(cur_rel_in)
            type_masks.append(cur_type_mask)
            # input channel
            cur_tok = cur_sent_in.copy()+cur_rel_in.copy()
            toks.append(cur_tok)

        # Form test loader
        dtset = MyDataset(toks, type_masks)
        collate_with_pad_id = partial(custom_collate, RE_PAD_TOKEN_ID=self.tokenizer.pad_token_id)
        test_loader = DataLoader(dataset=dtset, batch_size=512, collate_fn=collate_with_pad_id, shuffle=False)

        all_logits = []
        print("Start running model...")
        for i, data in enumerate(test_loader):
            #if i%2==0:
            #    print("\rEvaluating... %s%%" % (100*(i+1)/len(dev_loader)), end='', flush=True)
            input_ids, type_masks, att_masks = data
            input_ids, type_masks, att_masks = input_ids.cuda(self.gpu_id), type_masks.cuda(self.gpu_id), att_masks.cuda(self.gpu_id)
            self.model_lock.acquire()
            with torch.no_grad():
                _, logits = self.model(input_ids,\
                            enc_attention_mask=att_masks,\
                            token_type_ids=type_masks)
                # logits = torch.randn(len(input_ids),2).to(device)
                all_logits += nn.functional.softmax(logits, dim=1)[:,1].tolist()
            self.model_lock.release()
        assert(len(all_logits)==len(cand_rel))
        rel_logits = list(zip(cand_rel, all_logits))
        rel_logits.sort(reverse=True, key=lambda x:x[1])

        # print(rel_logits[:K])
        return rel_logits[:K]

    def extract_relation_with_cache(self, ques, m1, m2, cand_rel, K):
        # Wrapper function to implement RE cache
        # Fetch the cached score directly if one <ques, m1, m2, cand> is ran before
        # Otherwise, run model to get the scores and update RE cache

        # default case
        if len(cand_rel) == 0:
            if self.is_uri_all_relations:
                cand_rel = copy.deepcopy(self.all_relations_str)
            else:
                cand_rel = copy.deepcopy(self.all_relations)

        # get cached scores
        rel_scores, to_run_cand_rel = [], []
        for cur_cand in cand_rel:
            if ques+'_'+m1+'_'+m2+'_'+cur_cand in self.re_cache.keys():
                rel_scores.append(copy.deepcopy([cur_cand, self.re_cache[ques+'_'+m1+'_'+m2+'_'+cur_cand]]))
            else:
                to_run_cand_rel.append(cur_cand)
        
        # run model
        new_rel_scores = self.extract_relation(ques, m1, m2, to_run_cand_rel, len(to_run_cand_rel)) if len(to_run_cand_rel)!=0 else []
        
        # get all scores & sort
        rel_scores += new_rel_scores
        rel_scores.sort(reverse=True, key=lambda x:x[1])
        # update cache
        self.re_cache_lock.acquire()
        for new_rel, new_score in new_rel_scores:
            self.re_cache[ques+'_'+m1+'_'+m2+'_'+new_rel] = new_score
        self.re_cache_lock.release()
        
        return rel_scores[:K]
    
    def extract_relation_with_cache_uri(self, ques, m1, m2, cand_rel, K):
        # A wrapper of ``extract_relation_with_cache``, relations in cand_rel is in uri format
        # First transform each relation (in consistent with model training), then do raw RE

        # Process candidate relations in standard manner
        assert(self.is_uri_all_relations)
        cand_rel_strs, str_to_uri = [], copy.deepcopy(self.str_to_uri)
        for rel_uri in cand_rel:
            rel_info = resolve_rel(rel_uri)
            status = rel_info.get('success', True)
            if not status:  # invalid relation
                print("Invalid relation: {}".format(rel_uri))
                continue
            rel_str = rel_info['rel_type'] + ' ' + rel_info['rel_name']
            cand_rel_strs.append(rel_str)
            str_to_uri[rel_str] = rel_uri
        # Call raw RE
        res = self.extract_relation_with_cache(ques, m1, m2,cand_rel_strs, K)
        # Convert ack to uri
        for ix, (rel_str, score) in enumerate(res):
            res[ix] = [str_to_uri[rel_str], score]
        return res
    
    def flush_re_cache(self):
        # Write RE cache back to disk
        print("Start flushing cache, please DONOT CTRL+C!")
        if not self.re_cache_fn:
            return
        self.re_cache_lock.acquire()
        with open(self.re_cache_fn, 'w', encoding='utf-8') as fout:
            json.dump(self.re_cache, fout, ensure_ascii=False)
        self.re_cache_lock.release()
        print('Flushed RE cache to {}.'.format(self.re_cache_fn))


class KBQA_Server(socketserver.StreamRequestHandler):
    # overwrite the handler function to implement threading-server
    # each server-thread runs this function by default
    def handle(self):
        global re_model, neqg_model
        print("addr: ", self.client_address)
        # c.send("from server!".encode('utf-8'))

        while True:
            
            # resolve message length
            raw_msglen = recv_n_bytes(self.request, 4)
            if not raw_msglen:
                print("No message length.\nGoodbye to addr:", self.client_address)
                break
            # Read the message data
            msg_len = struct.unpack('>I', raw_msglen)[0]
            #msg = self.request.recv(102400000)
            msg = recv_n_bytes(self.request, msg_len)
            
            if not msg:
                print("Goodbye to addr:", self.client_address)
                # re_model.flush_re_cache()
                break

            msg = msg.decode()
            if len(msg) == 0:
                print("Got blank input! Stop current connection.")
                break
            
            # msg = json.loads(msg)
            try:
                msg = json.loads(msg)
            except Exception as e:
                print(traceback.format_exc())
                print(msg)
                rep_msg = "Syntax error encountered! Probably due to too long input!"
                print(rep_msg)
                self.request.send(json.dumps({'Error':rep_msg}).encode('utf-8'))
                break

            qtype = msg['type']
            # qtype, msg = msg.split('\t\t\t')

            if qtype == 'Flush_Cache':
                re_model.flush_re_cache()
                resp = {"Result": "cache flushed"}
                self.request.send(json.dumps(resp).encode('utf-8'))

            elif qtype == 'NC':
                sent = msg['sent']
                print("Recieved a Node Classification query.\nInput sentence: ", sent)
                try:
                    nodes = neqg_model.extract_node(sent, verbose=False)
                except AssertionError:
                    self.send_failure_message()
                    neqg_model.model_lock.release()
                    continue

                # nodes = [1,2,"dsf","第三方","dsf\ndf"]
                print("Nodes extracted:\n", nodes)
                nodes = {"Result":nodes}
                self.request.send(json.dumps(nodes).encode('utf-8'))
            
            elif qtype == 'QG':
                sent, use_beam_search = msg['sent'], msg['use_beam_search']
                num_beams, top_k = msg['num_beams'], msg['top_k']
                print("Received a Query Generation query.")
                print("Input sent: {}, use_beam_search={}, num_beams={}, topk={}".format(sent, use_beam_search, num_beams, top_k))

                try:
                    node_tags, pred_scores, pred_queries, pred_ptrs, nodes = \
                        neqg_model.generate_query_and_nodes(sent,\
                                                            use_beam_search=use_beam_search,\
                                                            num_beams=num_beams,\
                                                            top_k=top_k)
                except AssertionError:
                    self.send_failure_message()
                    neqg_model.model_lock.release()
                    continue

                results = {
                    "node_tags": node_tags,
                    "pred_scores": pred_scores,
                    "pred_queries": pred_queries,
                    "pred_ptrs": pred_ptrs,
                    "nodes": nodes
                }
                print(results)
                self.request.send(json.dumps(results).encode('utf-8'))
            
            else:
                ques,m1,m2,cand,K = msg['ques'],msg['m1'],msg['m2'],msg['cand_rel'],msg['K']
                # ques,m1,m2,cand,K = msg.split('\t\t')
                # cand = eval(cand)
                K = int(K)
                use_uri = msg.get('use_uri', False)
                print("Recieved a Relation Extraction query.\nInput: ", ques+" "+m1+" "+m2)
                
                try:
                    if not use_uri:
                        topk_rel = re_model.extract_relation_with_cache(ques, m1, m2, cand, K)
                    else:   # Relations in URI format
                        topk_rel = re_model.extract_relation_with_cache_uri(ques, m1, m2, cand, K)
                except AssertionError:
                    self.send_failure_message()
                    re_model.model_lock.release()
                    continue
                
                print("Relation scores:\n", topk_rel)
                topk_rel = {"Result":topk_rel}
                self.request.send(json.dumps(topk_rel).encode('utf-8'))
    
    def send_failure_message(self):
        exc = traceback.format_exc()
        print(exc)
        res = {
            'Success': 0,
            'Traceback': exc
        }
        self.request.send(json.dumps(res).encode('utf-8'))


def main():
    
    global re_model, neqg_model
    re_model_config_fn = './data/checkpoint_en/RE_Roberta/config.json'

    re_model = KBQA_Models(re_model_config_fn)
    # open server on ip:port
    host = 'localhost'
    port = 9305
    server = socketserver.ThreadingTCPServer((host, port), KBQA_Server)
    
    # Expand buffer size
    modify_socket_buffer_size(server.socket, 80000, 160000)
    print("After: rcv {}, snd {}.".format(server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF), server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)))
    
    server.serve_forever()


if __name__ == "__main__":

    main()
