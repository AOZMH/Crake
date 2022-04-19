from os import removedirs
import socket
import torch
from torch.utils.data import Dataset

# tag dict for Node-Extraction
tags = ('O','Eb','Ei','Vb','Vi','VTb','VTi','CTb','CTi','VLb','VLi','CLb','CLi','<START>','<STOP>')
tag2id = {tag:tid for tag,tid in zip(tags, range(len(tags)))}
id2tag = {tid:tag for tag,tid in tag2id.items()}

# tag dict for Query-Generation
qg_tags = ('<start>','<end>','entity','variable','literal','type')
qg_tag2id = {tag:tid for tag,tid in zip(qg_tags, range(len(qg_tags)))}
qg_id2tag = {tid:tag for tag,tid in qg_tag2id.items()}


def get_nodes(labels, ques):
    # given a seqence of node-label IDs, extract all nodes in list form. Sample input:
    # lables = [1, 2, 2, 2, 2, 0, 3, 4, 4, 4, 0, 0, 0]
    # ques = ['德', '国', '有', '哪', '些', '著', '名', '的', '汽', '车', '品', '牌', '？']
    # Sample return:
    # spans = [[0, 1], [8, 11]]
    # nodes = [['E', [0, 1]], ['V', [8, 11]]]
    # NOTE: assuming that NO PADDING is in labels & ques

    # preprocess before extracting all nodes
    state = 0
    for i,lab in enumerate(labels):
        if id2tag[lab] == 'O':
            state = 0
            continue
        if 'i' in id2tag[lab] and state==0:
            # the case when extra processing is needed
            # O->I
            state = 1
            if i>=1 and (ques[i-1] =='《' or ques[i-1] =="\"" or ques[i-1] =='“'):
                labels[i-1] = lab-1
                continue
            if i>=1 and (ques[i-1] =='》' or ques[i-1] =='”'):
                labels[i] = lab-1
                continue
            if i>=2 and id2tag[labels[i-2]]!='O':
                labels[i-1] = lab
                continue
            # start with I
            labels[i] = lab-1
        state = 1
    
    # extract nodes, vote to get the type of each node
    node_types = ("E","V","VT","CT","VL","CL")
    nodes = []  # list of nodes, each item is a pair of <node_type, [begin, end]>
    counts = [0]*6
    state = 0   # currently in a node or not
    pos = -1

    for i,lab in enumerate(labels+[0]):
        if id2tag[lab] == 'O':
            if state!=0:
                cur_type = node_types[counts.index(max(counts))]
                nodes.append([cur_type, [pos,i-1]])
                counts = [0]*6
            state = 0
        elif 'b' in id2tag[lab]:
            if state!=0:
                cur_type = node_types[counts.index(max(counts))]
                nodes.append([cur_type, [pos,i-1]])
                counts = [0]*6
            state = 1
            pos = i
            counts[(lab-1)//2] += 1
        elif id2tag[lab] not in ('<START>','<STOP>'):
            counts[(lab-1)//2] += 1
    
    # 对literal的附加规则：如果识别出的literal节点在一个引号区间内，则直接对齐到这个区间
    # 为后处理方便，规定引号本身并不包含在识别出的literal内
    if "\"" not in ques and "《" not in ques:    # 暂时只处理英文引号的对齐，故没有则跳过
        return nodes
    quote_ranges, punc_ranges = [],[]   # 先计算引号/书名号所覆盖的区域
    in_quote, in_punc = False, False
    for ix, tok in enumerate(ques):
        if tok=='\"':
            if not in_quote:
                quote_ranges.append([ix])
                in_quote = True
            else:
                quote_ranges[-1].append(ix)
                in_quote = False
        elif tok=='《':
            if not in_punc:
                punc_ranges.append([ix])
                in_punc = True
        elif tok=='》':
                punc_ranges[-1].append(ix)
                in_punc = False

    for ix, node in enumerate(nodes):
        cur_type, cur_span = node
        if "L" in cur_type: # 对齐literal
            l_pos, r_pos = cur_span
            for l_quote, r_quote in quote_ranges+punc_ranges:   # 若覆盖在某个引号/书名号区间内，则直接对齐；优先引号！
                if l_pos >= l_quote and r_pos <= r_quote:
                    aligned_range = [l_quote+1, r_quote-1]   # literal对齐区间不包括引号和书名号
                    nodes[ix][1] = aligned_range[:]
                    break
        if "E" in cur_type: # 对齐entity
            l_pos, r_pos = cur_span
            for l_quote, r_quote in punc_ranges+quote_ranges:   # 若覆盖在某个引号/书名号区间内，则直接对齐；优先书名号！
                if l_pos >= l_quote and r_pos <= r_quote:
                    aligned_range = [l_quote+1, r_quote-1] if ques[l_quote]=='\"' else [l_quote, r_quote]   # 对齐区间不包括引号，但包括书名号
                    nodes[ix][1] = aligned_range[:]
                    break
    
    # 由于对齐了，可能出现重复的node，需要去除；这里基于range进行去重，相同range的node取第一个
    unique_nodes, unique_ranges = [],[]
    for node in nodes:
        if node[1] not in unique_ranges:
            unique_nodes.append(node)
            unique_ranges.append(node[1])
    
    return unique_nodes


def load_all_relations(all_relations_fn):
    with open(all_relations_fn, 'r', encoding='utf-8') as f:
        all_rel = f.read().strip().split('\n')
    return all_rel


################ Helpers to form RE's dataloader ##################

class MyDataset(Dataset):
    def __init__(self, pairs, type_masks):
        self.pairs = pairs
        self.type_masks = type_masks

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.type_masks[idx]


def custom_collate(batch, RE_PAD_TOKEN_ID):
    transposed = list(zip(*batch))
    lst = []
    # transposed[0]: list of token ids of a question
    padded_seq = []
    type_mask = []
    att_mask = []
    max_seq_len = len(max(transposed[0], key=len))
    for seq, msk in zip(transposed[0], transposed[1]):
        padded_seq.append(seq + [RE_PAD_TOKEN_ID]*(max_seq_len-len(seq)))
        type_mask.append(msk + [1]*(max_seq_len-len(seq)))
        att_mask.append([1]*len(seq) + [0]*(max_seq_len-len(seq)))
    lst.append(torch.LongTensor(padded_seq))
    lst.append(torch.LongTensor(type_mask))
    lst.append(torch.FloatTensor(att_mask))
    
    return lst


################ Socket-related helpers ##################

def modify_socket_buffer_size(sock, snd_size=80000, rcv_size=160000):

    print("Modifying socket buffer size...")
    print("Before: rcv {}, snd {}.".format(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF), sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)))
    sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, snd_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcv_size)
    print("After: rcv {}, snd {}.".format(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF), sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)))


def recv_n_bytes(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if len(packet) == 0:
            return None
        data.extend(packet)
    return data


################ Convert URI relations to string format ##################

def resolve_rel(rel):
    if rel[:19] != 'http://dbpedia.org/':
        if rel == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
            return {
                'rel_name': 'type',
                'rel_type': ''
            }
        else:
            return {'success': False}
    rel = rel[19:]
    pos1 = rel.find('/')
    res = {
        'rel_type': rel[:pos1],
        'rel_name': rel[pos1+1:]
    }
    if res['rel_type'] not in ('property', 'ontology'):
        return {'success': False}
    if '/' in res['rel_name']:
        return {'success': False}
    return res
