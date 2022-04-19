# 基于 dbpedia lookup 的实体链接模块, 实现本地cache
import sys
sys.path.append('../')
import json
import copy
import requests
import xmltodict
from tqdm import tqdm
from pattern.text.en import singularize

from gstore_util.GstoreConnector import GstoreConnector


def get_pkubase_client(host="115.27.161.37", port=12345):
    # 注意这里的client默认cache地址不一样! 是为了查询同名实体存在性专用的cache
    user = "root"
    password = "123456"
    return GstoreConnector(host, port, user, password, use_cache=True, cache_fn="../gstore_util/entity_existence_cache.dat")


class dbpedia_lookup_entity_linker:
    
    def __init__(self, url='http://115.27.161.37:9273/lookup-application/api/search', cache_fn='../data/lookup_cache/cache.dat'):
        self.url = url
        self.cache_fn = cache_fn
        try:
            print('Loading dbpedia lookup entity linker cache from {}'.format(cache_fn))
            with open(cache_fn) as fin:
                self.cache = json.load(fin)
        except FileNotFoundError:
            print('Empty cache initiated.')
            self.cache = dict()
    
    def flush_cache(self):
        print('Flushing cache...')
        with open(self.cache_fn, 'w') as fout:
            json.dump(self.cache, fout)
        print('Lookup cache flushed to {}'.format(self.cache_fn))
    
    def get_label_results(self, mention, topk=5):
        # 获取dbpedia lookup label api返回的实体
        cur_key = '^'.join((mention, 'label'))
        # 直接走cache
        if cur_key in self.cache.keys():
            return copy.deepcopy(self.cache[cur_key][:topk])
        
        params = {
            "label": mention
        }
        r = requests.get(self.url, params=params)
        xmlparse = xmltodict.parse(r.content)
        # 无结果
        if xmlparse['results'] == None:
            res = []
        # 只有一个结果时需多包一层
        elif type(xmlparse['results']['result']) != list:
            res = [xmlparse['results']['result']]
        else:
            res = xmlparse['results']['result']

        ret = []
        for cur_res in res:
            cur_ret = {
                'resource': cur_res['resource'],
                'score': cur_res['score']
            }
            ret.append(cur_ret)
        
        # 刷新cache
        self.cache[cur_key] = copy.deepcopy(ret)
        return copy.deepcopy(ret[:topk])
    
    def get_query_results(self, mention, topk=5):
        # 获取dbpedia lookup label api返回的实体
        cur_key = '^'.join((mention, 'query'))
        # 直接走cache
        if cur_key in self.cache.keys():
            return copy.deepcopy(self.cache[cur_key][:topk])
        
        params = {
            "query": mention
        }
        r = requests.get(self.url, params=params)
        xmlparse = xmltodict.parse(r.content)
        # 无结果
        if xmlparse['results'] == None:
            res = []
        # 只有一个结果时需多包一层
        elif type(xmlparse['results']['result']) != list:
            res = [xmlparse['results']['result']]
        else:
            res = xmlparse['results']['result']

        ret = []
        for cur_res in res:
            cur_ret = {
                'resource': cur_res['resource'],
                'score': cur_res['score'],
                'refCount': cur_res.get('refCount', 0),
            }
            ret.append(cur_ret)
        
        # 刷新cache
        self.cache[cur_key] = copy.deepcopy(ret)
        return copy.deepcopy(ret[:topk])


# 无效relation, 添加同名实体时不考虑这些边
banned_relations = ["http://dbpedia.org/property/imageCaption", "http://dbpedia.org/property/caption", \
    "http://dbpedia.org/property/website", "http://dbpedia.org/property/shortDescription"]


def add_homo_entity(mention, kb_client):
    # 如果KB中有与mention完全相同的entity, 则可以直接添加一个
    # 若存在, 返回该实体uri; 否则返回None
    
    homo_entity = '<http://dbpedia.org/resource/{}>'.format(mention.strip().replace('  ', '_').replace(' ', '_'))
    sparql = '\t{' + homo_entity + ' ?x ?y.' + '} union {' + '?y ?x {}.'.format(homo_entity) + '} \n'
    sparql += '\tFILTER isURI({})\n\t'.format('?y')
    for banned_rel in banned_relations:
        sparql += "FILTER(?x != <{}>). ".format(banned_rel)
    sparql += "\n\tFILTER (regex(str(?x), \"dbpedia.org\") \n"
    sparql = 'select distinct ?x where {\n' + sparql + '\n}'
    #return sparql
    res = kb_client.query_with_cache('dbpedia_2016_04_core', 'json', sparql)
    res_dict = json.loads(res)
    if res_dict['StatusCode'] != 0:
        print(homo_entity + '\n' + sparql)
        return None
    if len(res_dict['results']['bindings']) > 0:
        return homo_entity[1:-1]
    else:
        return None


def add_homo_type(mention, kb_client, prefix='ontology'):
    # 如果KB中有与mention完全相同的type, 则可以直接添加一个
    # 若存在, 返回该type uri; 否则返回None
    
    homo_type = '<http://dbpedia.org/{}/{}>'.format(prefix, mention.strip().replace('  ', '_').replace(' ', '_'))
    sparql = '\t{' + homo_type + ' ?x ?y.' + '} union {' + '?y ?x {}.'.format(homo_type) + '} \n'
    sparql += '\tFILTER isURI({})\n\t'.format('?y')
    sparql += "\n\tFILTER (regex(str(?x), \"dbpedia.org\") || ?x = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>) \n"
    sparql += "\n\tFILTER (regex(str(?x), \"type\") \n"
    sparql = 'select distinct ?x where {\n' + sparql + '\n}'
    #return sparql
    res = kb_client.query_with_cache('dbpedia_2016_04_core', 'json', sparql)
    res_dict = json.loads(res)
    if res_dict['StatusCode'] != 0:
        print(homo_type + '\n' + sparql)
        return None
    if len(res_dict['results']['bindings']) > 0:
        return homo_type[1:-1]
    else:
        return None


def type_mention_format(mention):
    # type mention规范化, 转单数+首字母大写+去除空格
    sin_mention = singularize(mention)
    format_mention = ''.join([(sub_str[0].upper() + sub_str[1:] if sub_str else sub_str) for sub_str in sin_mention.strip().split(' ')])
    return format_mention


def judge_entity_name_match(mention, entity_name, question='', paraphrase_info_mapping={}, align=False):
    # 1. 判定entity在字面上是否与mention一致; 若是, 一般可提高其优先级
    # 2. 判断实体名后是否有 _(xxx) 释义
    #    若有则拆分出前半段作为实体名, 后半段作为语义信息辅助候选排序（若后半段的词出现于question中则提升优先级）
    
    if '_(' in entity_name:
        if not (entity_name.count('_(')==1 and entity_name[-1]==')'):
            return False, False
        real_name, extra_info = entity_name.split('_(')
        extra_info = extra_info[:-1]    # 去除右括号
        extra_info = extra_info.split('_')    # 分词
    else:
        real_name = entity_name
        extra_info = []
    
    # 判定1    
    formatted_name = real_name.replace('_', ' ').strip()
    if align:
        align_fnc = lambda x:x.replace(' ', '')
    else:
        align_fnc = lambda x:x
    is_match = align_fnc(formatted_name.lower()) == align_fnc(mention.strip().lower())    # TODO: 是否有进一步的判定？例如去除空格/特殊符号等, 需要diff study
    
    # 判定2
    is_proper = False
    lower_question = question.lower()
    for cur_info in extra_info:
        if cur_info.lower() in skip_info:
            continue
        if cur_info in real_name:
            continue
        if cur_info.lower() in lower_question:
            is_proper = True
            break
        if cur_info.lower() in paraphrase_info_mapping.keys():
            if paraphrase_info_mapping[cur_info.lower()].lower() in lower_question:
                is_proper = True
                break
    
    #if is_proper and is_match:
    #    print(mention, entity_name, '\n' + question)
    
    return is_match, is_proper


paraphrase_info_mapping = {}
skip_info = []


def link_entity(params):
    # 实体链接接口
    # ret: [{'resource': 'xxx', 'ref_count': 1, ...}, ...]
    mention = params['mention']
    question = params.get('question', '')
    kb_client = params['kb_client']
    linker = params['linker']
    label_topk, query_topk = params.get('label_topk', 5), params.get('query_topk', 5)
    # 首字母大写规范化
    add_format_results = params.get('add_format_results', False)
    format_mention = ' '.join([(sub_str[0].upper() + sub_str[1:] if sub_str else sub_str) for sub_str in mention.strip().split(' ')])
    # 判定实体名相同时是否不考虑空格等特殊字符
    name_align = params.get('name_align', False)

    # Step_1: 获取lookup结果
    res = linker.get_label_results(mention, label_topk)
    res += linker.get_query_results(mention, query_topk)
    if add_format_results and format_mention != mention:  # 可以添加规范化后的召回结果
        res += linker.get_label_results(format_mention, label_topk)
        res += linker.get_query_results(format_mention, query_topk)
    # Step_2: 提升相同实体优先级
    matched_proper_ents, matched_ents, other_ents = [], [], []
    for cur_res in res:
        # 获取实体名称
        if 'http://dbpedia.org/resource/' != cur_res['resource'][:28]:
            print('[Entity prefix error]', mention, cur_res)
        cur_entity_name = cur_res['resource'][28:]
        #if cur_entity_name.count('/') != 0:
        #    print(mention, cur_res) 
        # 判断实体名是否与mention一样
        is_match, is_proper = judge_entity_name_match(mention, cur_entity_name, question=question, paraphrase_info_mapping=paraphrase_info_mapping, align=name_align)
        if question == '':
            assert(not is_proper)
        if is_match:
            if is_proper:
                matched_proper_ents.append(cur_res)
            else:
                matched_ents.append(cur_res)
        else:
            other_ents.append(cur_res)
    # Step_3: 如果加入同名实体
    homo_entity = add_homo_entity(mention, kb_client)
    if not (homo_entity is None):
        matched_ents = [{'resource': homo_entity}] + matched_ents
    elif add_format_results and format_mention != mention:
        # 如果找不到同名实体, 将所有首字母大写再次寻找同名实体
        sub_homo_entity = add_homo_entity(format_mention, kb_client)
        if not (sub_homo_entity is None):
            matched_ents = [{'resource': sub_homo_entity}] + matched_ents
    # is_proper 优先级应该最高
    all_ents = matched_proper_ents + matched_ents + other_ents
    
    # 结果去重
    ents, uni_ents = set(), []
    for ix, cur_res in enumerate(all_ents):
        if cur_res['resource'] not in ents:
            ents.add(cur_res['resource'])
            uni_ents.append(cur_res)
    return uni_ents, res


def link_type(params):
    # 类型(type-node)链接接口
    # ret: [{'resource': 'xxx'}]
    mention = params['mention']
    kb_client = params['kb_client']
    train_type_mappings = params['train_type_mappings']

    # 直接从训练集type字典中取
    if mention in train_type_mappings:
        return [{'resource': train_type_mappings[mention][1]}]
    elif mention.lower() in train_type_mappings:
        return [{'resource': train_type_mappings[mention.lower()][1]}]
    # 若没有, 根据mention构造uri
    type_name = type_mention_format(mention)
    res = add_homo_type(type_name, kb_client)
    if res is None:     # 如果没有找到同名type, 则不考虑这个node, 应在之后处理中忽略当前triple
        return []
    return [{'resource': res}]

