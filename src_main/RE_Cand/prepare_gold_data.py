# 将标注完成的文件合并入train/test_full
# 处理数据，使之符合linked_results格式，以便运行RE_Candidate生成RE训练数据

import json
import sys
sys.path.append('../annotation/')
from preprocess import parse_sparql


def convert_to_linked_format(src, dst):
    with open(src) as fin:
        full_data = json.load(fin)
    
    output_data = []
    for dat in full_data:
        # find out target (select / ask) node
        node_set = dat['node_set']
        res = parse_sparql(dat['raw_sparql'])
        if res['target'] is None:
            target = ['None', 'ask', 'None']
        else:
            target = [res['target'], 'variable', node_set[res['target']]['mention']]

        cur_output = {
            '_id': dat['_id'],
            'Question': dat['question'],
            'Original sparql': dat['raw_sparql'],
            'Readable_sparql': dat['readable_sparql'],
            'Sparql_struct': [[target,],],
            'Sparql_scores': [-0.01,]
        }
        
        for triple in dat['triples']:
            # head
            if '?' in triple[0]:
                head = [triple[0], 'variable', node_set[triple[0]]['mention']]
            else:
                head = [triple[0], 'entity', node_set[triple[0]]['mention']]
            
            # tail (types may occur at tail)
            if '?' in triple[2]:
                tail = [triple[2], 'variable', node_set[triple[2]]['mention']]
            elif 'type' in triple[1].lower():
                tail = [triple[2], 'type', node_set[triple[2]]['mention']]
            else:
                tail = [triple[2], 'entity', node_set[triple[2]]['mention']]

            # add to Sparql_struct
            cur_output['Sparql_struct'][0].append([head, tail])
        output_data.append(cur_output)
    
    with open(dst, 'w') as fout:
        json.dump(output_data, fout, indent=4)


if __name__ == '__main__':
    
    #combine_annotation_to_full()
    #exit(0)
    
    dir_name = '../annotation/post_processed/'
    full_train_fn = dir_name + 'Train_full_data_annotated.json'
    full_test_fn = dir_name + 'Test_full_data_annotated.json'
    train_dst_fn = '../data/RE_gold/train_link.json'
    test_dst_fn = '../data/RE_gold/test_link.json'
    convert_to_linked_format(full_train_fn, train_dst_fn)
    convert_to_linked_format(full_test_fn, test_dst_fn)
