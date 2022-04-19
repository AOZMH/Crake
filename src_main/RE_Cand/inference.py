# conduct 1-hop inference over query candidates
# 1-hop means that relation candidates will be generated only when it has a determined path to some entities/types

import sys
sys.path.insert(0, "..")
sys.path.append("../../")

from gstore_util.GstoreConnector import GstoreConnector
from re_model_server.client import kbqa_client
from RE_Cand.re_cand_generate import parse_sparql

from func_timeout import func_set_timeout
import func_timeout
import json
import copy
from tqdm import tqdm
import math

# relations to be filtered out
banned_relations = ["http://dbpedia.org/property/imageCaption", "http://dbpedia.org/property/caption", \
    "http://dbpedia.org/property/website", "http://dbpedia.org/property/shortDescription"]

# debug mode flag
# set the flag to True will print additional debug info in standard output
debug = False

def get_pkubase_client(host="115.27.161.37", port=12345):
    user = "root"
    password = "123456"
    return GstoreConnector(host, port, user, password, use_cache=True, cache_fn="../gstore_util/pure_cache.dat")

def get_kbqa_client(host="115.27.161.60", port=9304):
    return kbqa_client(host, port)

# wrapper function that returns a *set* of results
@func_set_timeout(10000)
def fast_query_for_re(gc, sparql, key, kb='dbpedia_2016_04_core'):
    #print(sparql)
    # if debug:
    #     if sparql == 'SELECT DISTINCT ?r WHERE { FILTER isURI(?x). FILTER isURI(?uri). ?x <http://dbpedia.org/ontology/kingdom> <http://dbpedia.org/resource/Animal>. FILTER(?r != <http://dbpedia.org/property/imageCaption>). FILTER(?r != <http://dbpedia.org/property/caption>). FILTER(?r != <http://dbpedia.org/property/website>). FILTER(?r != <http://dbpedia.org/property/shortDescription>). FILTER (regex(str(?r), "dbpedia.org") || ?r = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>). ?x ?r ?uri. }':
    #         return set()
    if sparql == "blank":
        return set()
    res_json = gc.query_with_cache(kb, "json", sparql)
    #print(res_json)
    res_dict = json.loads(res_json)
    if res_dict['StatusCode'] != 0:
        return set()
    res_set = set()
    # print(res_dict['results']['bindings'])
    for item in res_dict['results']['bindings']:
        if key not in item:
            continue
        res_set.add(item[key]['value'])
    # print(res_dict)
    return res_set

# main class that stores sparql struct as a graph and conducts inference based on greedy search
class QueryGraph(object):
    def __init__(self, sparql_struct, question, topk=8):
        self.question = question
        self.sparql_struct = [self._to_tuple(sparql_struct[0])] + [(self._to_tuple(triple[0]), self._to_tuple(triple[1])) for triple in sparql_struct[1:]]
        self.V = set()
        self.reachable = set()
        self._initialize_vertices()
        # maintain the edge set in a beam form; containing subject, predicate and object.
        self.beams = [[]]
        self.scores = [0]
        self.bitset = [False for i in range(len(self.sparql_struct)-1)]        
        self.topk = topk
        self.query_history = [{"pos_cand": set(), "rev_cand": set()} for i in range(len(self.sparql_struct)-1)]

    def _to_tuple(self, list):
        return tuple(list)

    def _is_entity_or_type(self, node):
        if node[1] == 'entity' or node[1] == 'type':
            return True
        return False

    def _initialize_vertices(self):
        self.V.add(self.sparql_struct[0])
        for triple in self.sparql_struct[1:]:
            if self._is_entity_or_type(triple[0]):
                self.reachable.add(triple[0])
            if self._is_entity_or_type(triple[1]):
                self.reachable.add(triple[1])
            self.V.add(triple[0])
            self.V.add(triple[1])

    # input:    (str) relation, e.g. "?r"
    # output:   (str) filter string
    def _generate_filter_for_relation(self, relation):
        # expect "?r" as input
        query = ""
        for banned_rel in banned_relations:
            query += "FILTER(" + relation + " != <" + banned_rel + ">). "
        query += "FILTER (regex(str(" + relation + "), \"dbpedia.org\") || " + relation + \
        " = <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>). "
        return query

    # input:    (list) cur_triple, e.g. [["?uri", "variable", "river"], ["http://dbpedia.org/resource/Dead_Sea", "entity", "deadsea"]]
    #           (list) restricts, e.g. ["?uri", "<http://dbpedia.org/ontology/riverMouth>", "<http://dbpedia.org/resource/Dead_Sea>"] 
    # output:   (str) filter string that restricts variable nodes to be *uri*, i.e. not literal
    def _generate_filter_for_entities(self, cur_triple, restricts):
        variables = []
        if cur_triple[0][1] == "variable" and cur_triple[0][0] not in variables:
            variables.append(cur_triple[0][0])
        if cur_triple[1][1] == "variable" and cur_triple[1][0] not in variables:
            variables.append(cur_triple[1][0])
        for tri in restricts:
            if tri[0][0] == "?" and tri[0] not in variables:
                variables.append(tri[0])
            if tri[2][0] == "?" and tri[2] not in variables:
                variables.append(tri[2])
        filters = ["FILTER isURI("+var+"). " for var in variables]
        query = "".join(filters)
        return query

    # input:    (list) cur_triple, e.g. [["?uri", "variable", "river"], ["http://dbpedia.org/resource/Dead_Sea", "entity", "deadsea"]]
    #           (list) restricts, e.g. ["?uri", "<http://dbpedia.org/ontology/riverMouth>", "<http://dbpedia.org/resource/Dead_Sea>"] 
    # output:   ((str), (str)) positive and reverse query strings
    def _generate_query(self, cur_triple, restricts):
        sparql_prefix = "SELECT DISTINCT ?r WHERE { "
        sparql_suffix = "}"
        sparql_prefix += self._generate_filter_for_entities(cur_triple, restricts)
        if len(restricts) > 0:
            for prev_triple in restricts:
                sparql_prefix += prev_triple[0] + " " + prev_triple[1] + " " + prev_triple[2] + ". "
        sparql_prefix += self._generate_filter_for_relation("?r")
        pos_query = sparql_prefix + cur_triple[0][0] + " ?r " + cur_triple[1][0] + ". " + sparql_suffix
        rev_query = sparql_prefix + cur_triple[1][0] + " ?r " + cur_triple[0][0] + ". " + sparql_suffix
        return (pos_query, rev_query)

    # conducts inference and stores results at self.beams, self.scores, (self.query_history if save_result is True)
    # note: save_result:    whether stores intermediate relation candidates
    #       pruning:        whether enables pruning. If set to True, empty relation list will not be considered when generating beams    
    def generate_re_result(self, gc, client, save_result=False, pruning=False):
        # 2 cases when returning:
        # 1) all relations are generated successfully;
        # 2) some triples can't generate relations.
        # in 2), (maybe) we should discard these triples.
        if sum(self.bitset) == len(self.bitset):
            # all triples have generated one's predicate, end of recursion
            return
        for index, triple in enumerate(self.sparql_struct[1:]):
            if self.bitset[index]:
                continue
            if triple[0] in self.reachable or triple[1] in self.reachable:
                # assume that we find a triple that hasn't generated predicate and is able to generate.
                new_beams = []
                new_scores = []
                for beam, beam_score in zip(self.beams, self.scores):
                    pos_query, rev_query = self._generate_query(triple, beam)
                    pos_rel = set()
                    rev_rel = set()
                    # try:
                    #     if debug:
                    #         print("Current query:", pos_query)
                    #     pos_rel = fast_query_for_re(gc, pos_query, "r")
                    # except func_timeout.exceptions.FunctionTimedOut:
                    #     print("TIMEOUT error in sparql query:\t", pos_query)
                    # try:
                    #     if debug:
                    #         print("Current query:", rev_query)
                    #     rev_rel = fast_query_for_re(gc, rev_query, "r")
                    # except func_timeout.exceptions.FunctionTimedOut:
                    #     print("TIMEOUT error in sparql query:\t", rev_query)
                    pos_rel = fast_query_for_re(gc, pos_query, "r")
                    rev_rel = fast_query_for_re(gc, rev_query, "r")
                    if debug:
                        print("# of positive candidate(s):\t", len(pos_rel))
                        print("# of reverse candidate(s):\t", len(rev_rel))
                    for rel in pos_rel:
                        try:
                            validation_check(rel)
                        except ValueError:
                            print("Unexpected relation ", rel, " in SPARQL:\t", pos_query)
                    for rel in rev_rel:
                        try:
                            validation_check(rel)
                        except ValueError:
                            print("Unexpected relation ", rel, " in SPARQL:\t", rev_query)
                    if save_result:
                        self.query_history[index]["pos_cand"] = self.query_history[index]["pos_cand"] | copy.deepcopy(pos_rel)
                        self.query_history[index]["rev_cand"] = self.query_history[index]["rev_cand"] | copy.deepcopy(rev_rel)
                    pos_rel = list(pos_rel)
                    rev_rel = list(rev_rel)
                    # important rule:
                    # make sure *node <type> ?x.* triple not exists!
                    if not self._is_entity_or_type(triple[0]) and "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in rev_rel:
                        # triple[0] is variable
                        rev_rel.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    if not self._is_entity_or_type(triple[1]) and "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in pos_rel:
                        # triple[1] is variable
                        pos_rel.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")    
                    # res form:
                    # [[rel, score],...]
                    if not (pruning and len(pos_rel) == 0):
                        pos_res = client.extract_relation(self.question, triple[0][2], triple[1][2], self.topk, pos_rel, use_uri=True)['Result']
                        for cand in pos_res:
                            if cand[0] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and not self._is_entity_or_type(triple[1]):
                                # double check
                                continue
                            new_beam = copy.deepcopy(beam)
                            new_beam.append([triple[0][0], "<"+cand[0]+">", triple[1][0]])
                            # count the log prob <=> maximize the total prob
                            # no heurestic rule of more weight for positive relation 
                            new_beam_score = beam_score + math.log(cand[1])
                            new_beams.append(new_beam)
                            new_scores.append(new_beam_score)
                    if not (pruning and len(rev_rel) == 0):
                        rev_res = client.extract_relation(self.question, triple[1][2], triple[0][2], self.topk, rev_rel, use_uri=True)['Result']
                        for cand in rev_res:
                            if cand[0] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and not self._is_entity_or_type(triple[0]):
                                continue
                            new_beam = copy.deepcopy(beam)
                            new_beam.append([triple[1][0], "<"+cand[0]+">", triple[0][0]])
                            # count the log prob <=> maximize the total prob
                            new_beam_score = beam_score + math.log(cand[1])
                            new_beams.append(new_beam)
                            new_scores.append(new_beam_score)
                if len(new_beams) == 0:
                    # to be implemented
                    # raise ValueError("why?")
                    self.bitset[index] = True
                # update info
                else:
                    self.reachable.add(triple[0])
                    self.reachable.add(triple[1])
                    self.bitset[index] = True
                    # sort
                    zipped_beams_and_scores = list(zip(new_beams, new_scores))
                    zipped_beams_and_scores = sorted(zipped_beams_and_scores, key=lambda x: x[1], reverse=True)
                    new_beams, new_scores = list(zip(*zipped_beams_and_scores))
                    new_beams, new_scores = list(new_beams), list(new_scores)
                    # need to reduce the beam size to topk
                    new_beams = new_beams if len(new_beams) <= self.topk else new_beams[:self.topk]
                    new_scores = new_scores if len(new_scores) <= self.topk else new_scores[:self.topk]
                    # update info, go into recursion
                    self.beams = new_beams
                    self.scores = new_scores
                
                self.generate_re_result(gc, client, save_result, pruning)
        return

# add "<>" to entities            
def sparql_struct_preprocess(sparql_struct):
    for triple in sparql_struct[1:]:
        if triple[0][1] != "variable":
            triple[0][0] = "<" + triple[0][0] + ">"
        if triple[1][1] != "variable":
            triple[1][0] = "<" + triple[1][0] + ">"
    return sparql_struct

def validation_check(rel):
    if "dbpedia.org" in rel:
        return
    if rel == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
        return
    raise ValueError("Unexpected relation :\t"+rel)

def gen_re_result(input_path, output_path, topk=8):
    """
    since it is impossible to generate RE candidates in multi-hop situation, 
    we have to utilize intermediate RE inference results to help reduce the
    search space. Thus, we combine RE candidate generation and RE result inference
    in one process.
    ==============================================================================
    Expected output format:
    {'Candidate triples':   [[triples],...(# of sparql structs)...,[...]], 
    'Candidate scores':     same size as above}
    Note:
    For ease of processing, entities in 'Sparql_struct' are enclosed by <>.
    ==============================================================================
    input_path:     the path of input data.
    output_path:    the path of output data.
    topk:           the hyperparameter defining the beam width.
    """
    # setup connectors to servers
    gc = get_pkubase_client(port=9275)
    client = get_kbqa_client()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for index, entry in enumerate(tqdm(data)):
        question = entry["Question"]
        candidate_structs = entry["Sparql_struct"]
        entry["Candidate triples"] = []
        entry["Candidate scores"] = []
        for sparql_struct in candidate_structs:
            # preprocess: add <> to all entity/type!
            sparql_struct = sparql_struct_preprocess(sparql_struct)
    
            qg = QueryGraph(sparql_struct, question, topk)
            qg.generate_re_result(gc, client, pruning=True)
            
            entry["Candidate triples"].append(copy.deepcopy(qg.beams))
            entry["Candidate scores"].append(copy.deepcopy(qg.scores))
        
    #gc.update_cache() 

    print("Finish processing and generating candidates.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Results are successfully wrote back to %s." % (output_path))

