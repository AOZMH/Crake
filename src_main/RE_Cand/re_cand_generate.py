import sys
sys.path.append("..")

from gstore_util.GstoreConnector import GstoreConnector

from func_timeout import func_set_timeout
import func_timeout
import json
import copy
from tqdm import tqdm

banned_relations = set(["http://dbpedia.org/property/imageCaption", "http://dbpedia.org/property/caption", \
    "http://dbpedia.org/property/website", "http://dbpedia.org/property/shortDescription"])


def get_pkubase_client(host="115.27.161.37", port=12345):
    user = "root"
    password = "123456"
    return GstoreConnector(host, port, user, password, use_cache=True, cache_fn="../gstore_util/cache.dat")

def parse_sparql(sparql): 
    """
    adapt from preprocess.py

    feature:
        1) no ?r;
        2) no literal (e.g. "123")
        3) no union
        4) maximum 3 triple
    """
    lpos, rpos = sparql.find("{"), sparql.find("}")
    header, triples, lower_spq = sparql[:lpos], sparql[lpos+1:rpos]+" ", sparql.lower()
    # parsed result dict
    parsed_result = {"raw_sparql": sparql}
    
    # parsing header, i.e. the SELECT or ASK information
    if "ask" in lower_spq[:10]:
        parsed_result["type"] = "ask"
        parsed_result["target"] = None
    else:
        parsed_result["type"] = "select"
        target = header.replace("select", "").replace("SELECT", "").replace("distinct", "").replace("DISTINCT", "").replace("where", "").replace("WHERE", "").strip()
        parsed_result["target"] = target
        pos = target.find("(")
        if pos != -1:    # function in target node, i.e. COUNT for LCQUAD
            func = target[:pos]
            var = target[pos+1:-1]
            parsed_result["function"] = func
            parsed_result["target"] = var
        else:
            parsed_result["function"] = None
            parsed_result["target"] = target
    
    # parsing query body, i.e. each triple
    # NO need to parse filter/order/limit/group tokens for LCQUAD
    cur_triple, all_triples, ix = [],[],0
    while ix < len(triples):
        if triples[ix] == '?':    # variable
            end_pos1, end_pos2 = triples.find(' ', ix+1), triples.find('.', ix+1)
            end_pos2 = 10000 if end_pos2 == -1 else end_pos2
            end_pos = min(end_pos1, end_pos2)
            cur_triple.append(triples[ix:end_pos])
            if len(cur_triple) == 3:
                all_triples.append(cur_triple)
                cur_triple = []
            ix = end_pos
        elif triples[ix] == '<':    # entity
            end_pos = triples.find('>', ix+1)
            cur_triple.append(triples[ix:end_pos+1])
            if len(cur_triple) == 3:
                all_triples.append(cur_triple)
                cur_triple = []
            ix = end_pos
        ix += 1
    
    # form a readable SPARQL by revealing entity names
    readable_sparql = header[:] + "{ "
    # also save the node set for annotation of node-mention alignment
    node_set = dict()
    for triple in all_triples:
        for ix, node in enumerate(triple):
            if node[0] == "?":
                readable_sparql += node + " "
            else:
                entity_name = node[node.rfind("/")+1:]
                readable_sparql += "<" + entity_name + "> "
        readable_sparql += "."
    readable_sparql += " }"
    
    parsed_result["triples"] = all_triples
    parsed_result["readable_sparql"] = readable_sparql
    return parsed_result

@func_set_timeout(20)
def fast_query_for_re(gc, sparql, key, kb='dbpedia_2016_04_core'):
    #print(sparql)
    if sparql == "blank":
        return set()
    res_json = gc.query_with_cache(kb, "json", sparql)
    #print(res_json)
    res_dict = json.loads(res_json)
    if res_dict['StatusCode'] != 0:
        return set()
    res_set = set()
    for item in res_dict['results']['bindings']:
        res_set.add(item[key]['value'])
    return res_set

# def test():
    # gc = get_pkubase_client()
    # rel_dict = dict()
    # with open("../data/RE_gold/test_link.json", "r") as f:
    #     data = json.load(f)
    # for idx, d in enumerate(data):
    #     print(idx)
    #     rel_list = gen_re_cand_1hop(gc, d["Original sparql"])
    #     for rl in rel_list:
    #         for r in rl["pos_cand"]:
    #             if r in rel_dict:
    #                 rel_dict[r] += 1
    #             else:
    #                 rel_dict[r] = 1
    #         for r in rl["rev_cand"]:
    #             if r in rel_dict:
    #                 rel_dict[r] += 1
    #             else:
    #                 rel_dict[r] = 1
    # ordered_dict = sorted(rel_dict.items(), key=lambda x:x[1], reverse=True)    
    # print(ordered_dict[:100])

def gen_re_cand_1hop(gc, sparql):
    sparql_struct = parse_sparql(sparql)
    rel_list = []
    triples_wo_r = [[tri[0], tri[2]] for tri in sparql_struct["triples"]]
    for idx, triple in enumerate(sparql_struct["triples"]):
        h, r, t = triple[0], triple[1], triple[2]
        var_cnt = h.startswith("?") + t.startswith("?")
        if var_cnt != 2:
        # single/zero variable case
            pos_query_sparql = "SELECT DISTINCT ?r WHERE { " + h + " ?r " + t + ". " + \
                gen_filter_regex("?r") + gen_filter_banned_relation("?r", banned_relations) + "}"
            rev_query_sparql = "SELECT DISTINCT ?r WHERE { " + t + " ?r " + h + ". " + \
                gen_filter_regex("?r") + gen_filter_banned_relation("?r", banned_relations) + "}"
            pos_rel = set()
            rev_rel = set()
            try:                
                pos_rel = fast_query_for_re(gc, pos_query_sparql, "r")
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", pos_query_sparql)
            try:
                rev_rel = fast_query_for_re(gc, rev_query_sparql, "r")
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", rev_query_sparql)
            gold_rel = [r[1:-1]]
            rel_list.append({
                "gold": gold_rel,
                "pos_cand": list(pos_rel),
                "rev_cand": list(rev_rel),
            })
        else:
            restricts = sparql_struct["triples"]
            restrict_query = ""
            for res in restricts:
                restrict_query += res[0] + res[1] + res[2] + "."
            sparql_prefix = "SELECT DISTINCT ?r WHERE { "
            sparql_suffix = "}"
            # print(restrict_query)
            # print(gen_filter_regex("?r"))
            # print(gen_filter_banned_relation("?r", banned_relations))
            pos_query_sparql = sparql_prefix+h+"?r"+t+". "+restrict_query+\
                gen_filter_regex("?r")+gen_filter_banned_relation("?r", banned_relations)+sparql_suffix
            rev_query_sparql = sparql_prefix+t+"?r"+h+". "+restrict_query+\
                gen_filter_regex("?r")+gen_filter_banned_relation("?r", banned_relations)+sparql_suffix
            pos_rel = set()
            rev_rel = set()
            try:                
                pos_rel = fast_query_for_re(gc, pos_query_sparql, "r")
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", pos_query_sparql)
            try:
                rev_rel = fast_query_for_re(gc, rev_query_sparql, "r")
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", rev_query_sparql)
            gold_rel = [r[1:-1]]
            rel_list.append({
                "gold": gold_rel,
                "pos_cand": list(pos_rel),
                "rev_cand": list(rev_rel),
            })
    return rel_list

def gen_re_cand(gc, sparql):
    sparql_struct = parse_sparql(sparql)
    rel_list = []
    triples_wo_r = [[tri[0], tri[2]] for tri in sparql_struct["triples"]]
    for idx, triple in enumerate(sparql_struct["triples"]):
        h, r, t = triple[0], triple[1], triple[2]
        var_cnt = h.startswith("?") + t.startswith("?")
        if var_cnt != 2:
        # single/zero variable case
            pos_query_sparql = "SELECT DISTINCT ?r WHERE { " + h + " ?r " + t + ". " + \
                gen_filter_regex("?r") + " }"
            rev_query_sparql = "SELECT DISTINCT ?r WHERE { " + t + " ?r " + h + ". " + \
                gen_filter_regex("?r") + " }"
            pos_rel = set()
            rev_rel = set()
            try:                
                pos_rel = fast_query_for_re(gc, pos_query_sparql, "r")
                pos_rel = pos_rel - banned_relations
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", pos_query_sparql)
            try:
                rev_rel = fast_query_for_re(gc, rev_query_sparql, "r")
                rev_rel = rev_rel - banned_relations
            except func_timeout.exceptions.FunctionTimedOut:
                print("TIMEOUT in sparql query:", rev_query_sparql)
            gold_rel = [r[1:-1]]
            rel_list.append({
                "gold": gold_rel,
                "pos_cand": list(pos_rel),
                "rev_cand": list(rev_rel),
            })
        else:
            restricts = triples_wo_r[:idx] + triples_wo_r[idx+1:]
            restricts = gen_bidirectional_product(restricts)
            sparql_prefix = "SELECT DISTINCT ?r WHERE { "
            sparql_suffix = "}"
            pos_query_sparqls = [sparql_prefix+h+" ?r "+t+". "+gen_restrict_query_from_list(restrict_list)+\
                gen_filter_regex("?r")+sparql_suffix for restrict_list in restricts]
            rev_query_sparqls = [sparql_prefix+t+" ?r "+h+". "+gen_restrict_query_from_list(restrict_list)+\
                gen_filter_regex("?r")+sparql_suffix for restrict_list in restricts]
            pos_rel = set()
            rev_rel = set()
            for q in pos_query_sparqls:
                try:
                    r_set = fast_query_for_re(gc, q, "r")
                    pos_rel = pos_rel | r_set
                except func_timeout.exceptions.FunctionTimedOut:
                    print("TIMEOUT in sparql query:", q)
            for q in rev_query_sparqls:
                try:
                    r_set = fast_query_for_re(gc, q, "r")
                    rev_rel = rev_rel | r_set
                except func_timeout.exceptions.FunctionTimedOut:
                    print("TIMEOUT in sparql query:", q)
            gold_rel = [r[1:-1]]
            pos_rel = pos_rel - banned_relations
            rev_rel = rev_rel - banned_relations
            rel_list.append({
                "gold": gold_rel,
                "pos_cand": list(pos_rel),
                "rev_cand": list(rev_rel),
            })
    return rel_list

def main():
    gc = get_pkubase_client(port=9275)
    with open("../data/RE_gold/test_link.json", "r") as f:
        test_data = json.load(f)
    with open("../data/RE_gold/train_link.json", "r") as f:
        train_data = json.load(f)
    for example in tqdm(test_data, desc="TEST"):
        rel_list = gen_re_cand_1hop(gc, example["Original sparql"])
        example["Relation list"] = rel_list
    for example in tqdm(train_data, desc="TRAIN"):
        rel_list = gen_re_cand_1hop(gc, example["Original sparql"])
        example["Relation list"] = rel_list
    with open("../data/RE_gold/test_re_1hop.json", "w") as f:
        json.dump(test_data, f, indent=4)
    with open("../data/RE_gold/train_re_1hop.json", "w") as f:
        json.dump(train_data, f, indent=4)
    

def check():
    pass
    # check gold == None
    # check filter relation

def gen_bidirectional_product(triple_list):
    if len(triple_list) == 0:
        return []
    elif len(triple_list) == 1:
        pos = triple_list[-1]
        rev = [pos[1], pos[0]]
        return [pos, rev]
    else:
        pos = triple_list[-1]
        rev = [pos[1], pos[0]]
        other_cand = gen_bidirectional_product(triple_list[:-1])
        cand = [[pos]+[o] for o in other_cand] + [[rev]+[o] for o in other_cand]
        return cand

def gen_restrict_query_from_list(triple_list):
    query = ""
    for idx, tri in enumerate(triple_list):
        query += tri[0] + " ?r" + str(idx) + " " + tri[1] + ". "
        # if tri[0][0] == "?":
        #     query += "FILTER regex(str(" + tri[0] + "), \"dbpedia.org\"). "
        # if tri[1][0] == "?":
        #     query += "FILTER regex(str(" + tri[1] + "), \"dbpedia.org\"). "
        query += gen_filter_regex("?r"+str(idx))
        # query += gen_filter_banned_relation("?r"+str(idx), banned_relations)
    return query

def gen_filter_regex(relation):
    query = "{FILTER regex(str(" + relation + "), \"dbpedia.org\"). } UNION {FILTER(" + relation + \
        " = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>). }"
    return query

def gen_filter_banned_relation(r, banned_relations):
    query = ""
    for relation in banned_relations:
        query += "FILTER(" + r + "!= <" + relation + ">). "
    return query
        
if __name__ == '__main__':
    # # cur_sparql = 'SELECT DISTINCT COUNT(?uri) WHERE { <http://dbpedia.org/resource/Channel_District> <http://dbpedia.org/ontology/state> ?uri }'
    # # # print(parse_sparql(cur_sparql))
    # # # test()
    
    # # # print(gen_re_cand(gc, cur_sparql))

    # gc = get_pkubase_client()
    # sparql = 'SELECT DISTINCT ?r WHERE { <http://dbpedia.org/resource/Channel_District> ?r ?uri. {FILTER(?r = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)} UNION {FILTER regex(str(?r), "dbpedia")}.}'
    # print(fast_query_for_re(gc, sparql, "r"))

    # # print(gc.query('dbpedia_2016_04_core', 'json', sparql))
    # # # triple_list = [[["a", 1], ["b", 2]], [["c", 3], ["d", 4]]]
    # # # print(gen_bidirectional_product(triple_list))

    main()

    # with open("../data/RE_gold/test_re_1hop.json", "r") as f:
    #     test_data = json.load(f)
    # with open("../data/RE_gold/train_re_1hop.json", "r") as f:
    #     train_data = json.load(f)
    # with open("../data/RE_gold/train_re_1hop.json", "w") as f:
    #     json.dump(train_data, f, indent=4)
    # with open("../data/RE_gold/test_re_1hop.json", "w") as f:
    #     json.dump(test_data, f, indent=4)
