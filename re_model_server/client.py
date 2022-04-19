import socket
import json
import struct


class kbqa_client():
    
    def __init__(self, host_ip, port):
        super(kbqa_client, self).__init__()
        self.conn = socket.socket()
        self.host = host_ip
        self.port = port
        self.conn.connect((self.host, self.port))
    
    def send_json_msg(self, msg):
        # Prepend message length for each transmission
        msg_to_send = json.dumps(msg).encode('utf-8')
        msg_to_send = struct.pack('>I', len(msg_to_send)) + msg_to_send
        self.conn.send(msg_to_send)

    def extract_nodes(self, ques):
        # Extract all nodes in ques (a string)
        # Returns a list of nodes, each resembles: [node, type_of_node, position_in_ques]

        msg = {}
        msg['type'] = 'NC'
        msg['sent'] = ques
        
        self.send_json_msg(msg)
        nodes = self.conn.recv(51200).decode()
        return json.loads(nodes)

    def extract_relation(self, ques, mention1, mention2, K, cand_rel=[], use_uri=False):
        # Select the most probable relation between mention 1&2 in ques
        # Selection is done from candidate relations.
        # If candidates are not provided, the selection is performed on server's default all relations
        
        msg = {}
        msg['type'] = 'RE'
        msg['ques'] = ques
        msg['m1'] = mention1
        msg['m2'] = mention2
        msg['cand_rel'] = cand_rel
        msg['K'] = K
        msg['use_uri'] = use_uri
        
        self.send_json_msg(msg)
        topk_rel = self.conn.recv(1000000).decode()
        return json.loads(topk_rel)

    def generate_query_and_nodes(self, ques, use_beam_search=True, num_beams=4, top_k=4):
        # Extract nodes and generate node_tag & pointer sequence of a given sentence
        # Set up num_beams, top_k to customize beam_search

        msg = {
            "type": "QG",
            "sent": ques,
            "use_beam_search": use_beam_search,
            "num_beams": num_beams,
            "top_k": top_k
        }

        self.send_json_msg(msg)
        results = self.conn.recv(1000000).decode()
        return json.loads(results)

    def flush_server_cache(self):
        msg = {"type": "Flush_Cache"}
        self.send_json_msg(msg)
        results = self.conn.recv(10000).decode()
        print("RE cache flushed.\n", json.loads(results), sep='')

    def stop_client(self):
        self.conn.close()
    
    def reconnect(self):
        self.conn.close()
        self.conn = socket.socket()
        self.conn.connect((self.host, self.port))


def main(client, ques = "毕业于张旻昊所就读的高中的校长之前任教的高中，人称“小zang”的同学所处实验室小组的男性实习生是谁？"):
    nodes = client.extract_nodes(ques)
    print(nodes)


def lcquad_re_test():

    host = 'localhost'
    host = '115.27.161.60'
    port = 9304
    client = kbqa_client(host, port)

    ques = "Which battles were fought under the president when Chung Won Shik was the prime minister?"
    m1 = "president"
    m2 = "Chung Won Shik"
    rels = [
        "http://dbpedia.org/property/primeminister",
        "http://dbpedia.org/ontology/primeMinister",
        "http://dbpedia.org/ontology/predecessor",
        "http://dbpedia.org/property/predecessor"
    ]
    K = 20
    res = client.extract_relation_from_uri(ques,m1,m2,K,rels)
    print(res)
    exit(0)


if __name__ == "__main__":

    lcquad_re_test()

    host = 'localhost'
    host = '115.27.161.60'
    port = 9303
    client = kbqa_client(host, port)

    res = client.generate_query_and_nodes("《富春山居图》是谁的作品？")
    print(res)
    client.stop_client()
    exit(0)

    for i in range(100):
        main(client, "《富春山居图》是谁的作品？")
        #exit(0)
        
        ques = "由梁翘柏作曲，李焯雄填词的歌曲有哪些？"
        m1 = "歌曲"
        m2 = "李焯雄"
        # rels = ["填词","作词","谱曲","编曲","词"]
        rels = []
        K = 20
        res = client.extract_relation(ques,m1,m2,K,rels)
        print(res)

        main(client, "由梁翘柏作曲，李焯雄填词的歌曲有哪些？")
        main(client)
    client.stop_client()
