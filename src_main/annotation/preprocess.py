
def parse_sparql(sparql): 
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
            cur_triple.append(triples[ix+1:end_pos])
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
            if ix != 1:    # relations are not nodes
                node_set[node] = {
                    "name": entity_name if node[0] != "?" else node,
                    "mention": "",
                }
        readable_sparql += "."
    readable_sparql += " }"
    
    parsed_result["triples"] = all_triples
    parsed_result["readable_sparql"] = readable_sparql
    parsed_result["node_set"] = node_set
    return parsed_result


