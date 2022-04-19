"""
# Filename: GStoreConnector.py
# Last Modified: 2019-5-15 18:10
# Description: http api for python
"""

import sys
import requests
import json
#from func_timeout import func_set_timeout

version = sys.version[0]
if version == '3':
    from urllib import parse

defaultServerIP = "127.0.0.1"
defaultServerPort = "9000"

class GstoreConnector:
    def __init__(self, ip, port, username, password, use_cache=False, cache_fn='./pkubase_util/data/cache.dat'):
        if (ip == "localhost"):
            self.serverIP = defaultServerIP
        else:
            self.serverIP = ip
        self.serverPort = port
        self.Url = "http://" + self.serverIP + ":" + str(self.serverPort)
        self.username = username
        self.password = password
        self.use_cache = use_cache
        if use_cache:
            self.cache_fn = cache_fn
            try:
                print("Loading cache file from {}...".format(cache_fn))
                with open(cache_fn, 'r', encoding='utf-8') as fin:
                    self.cache = json.load(fin)
                print("Cache loaded.")
            except FileNotFoundError:
                self.cache = dict()
                print("Empty cache initiated")
   
    def UrlEncode(self, s):
        ret = ""
        if version == '2':
            for i in range(len(s)):
                c = s[i]
                if ((ord(c)==42) or (ord(c)==45) or (ord(c)==46) or (ord(c)==47) or (ord(c)==58) or (ord(c)==95)):
                    ret += c
                elif ((ord(c)>=48) and (ord(c)<=57)):
                    ret += c
                elif ((ord(c)>=65) and (ord(c)<=90)):
                    ret += c
                elif ((ord(c)>=97) and (ord(c)<=122)):
                    ret += c
                elif (ord(c)==32):
                    ret += '+'
                elif ((ord(c)!=9) and (ord(c)!=10) and (ord(c)!=13)):
                    ret += "{}{:X}".format("%", ord(c))
        elif version == '3':
            ret = parse.quote(s)
        return ret

    def Get(self, strUrl):
        r = requests.get(self.Url + self.UrlEncode(strUrl))
        r.encoding = "utf-8"
        return r.text

    def Post(self, strUrl, strPost):
        r = requests.post(self.Url + self.UrlEncode(strUrl), strPost)
        r.encoding = "utf-8"
        return r.text

    def fGet(self, strUrl, filename):
        r = requests.get(self.Url + self.UrlEncode(strUrl), stream=True)
        r.encoding = "utf-8"
        with open(filename, 'wb') as fd:
            for chunk in r.iter_content(4096):
                fd.write(chunk)
        return

    def fPost(self, strUrl, strPost, filename):
        r = requests.post(self.Url + self.UrlEncode(strUrl), strPost, stream=True)
        r.encoding = "utf-8"
        with open(filename, 'wb') as fd:
            for chunk in r.iter_content(4096):
                fd.write(chunk)
        return

    def build(self, db_name, rdf_file_path, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=build&db_name=" + db_name + "&ds_path=" + rdf_file_path + "&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/build"
            strPost = '{\"db_name\": \"' + db_name + '\", \"ds_path\": \"' + rdf_file_path + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def load(self, db_name, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=load&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/load"
            strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res
    
    def unload(self, db_name, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=unload&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/unload"
            strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)             
        return res

    def user(self, type, username2, addition, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=user&type=" + type + "&username1=" + self.username + "&password1=" + self.password + "&username2=" + username2 + "&addition=" +addition
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/user"
            strPost = '{\"type\": \"' + type + '\", \"username1\": \"' + self.username + '\", \"password1\": \"' + self.password + '\", \"username2\": \"' + username2 + '\", \"addition\": \"' + addition + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def showUser(self, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=showUser&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/showUser"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def query(self, db_name, format, sparql, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=query&username=" + self.username + "&password=" + self.password + "&db_name=" + db_name + "&format=" + format + "&sparql=" + sparql
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/query"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\", \"db_name\": \"' + db_name + '\", \"format\": \"' + format + '\", \"sparql\": \"' + sparql + '\"}'
            res = self.Post(strUrl, strPost)
        return res
    
    # @func_set_timeout(20)
    def query_with_cache(self, db_name, format, sparql, request_type='GET'):

        if self.use_cache and (db_name + '__' + sparql) in self.cache.keys():
            return self.cache[(db_name + '__' + sparql)]

        if request_type == 'GET':        
            strUrl = "/?operation=query&username=" + self.username + "&password=" + self.password + "&db_name=" + db_name + "&format=" + format + "&sparql=" + sparql
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/query"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\", \"db_name\": \"' + db_name + '\", \"format\": \"' + format + '\", \"sparql\": \"' + sparql + '\"}'
            res = self.Post(strUrl, strPost)
        
        if self.use_cache:
            self.cache[(db_name + '__' + sparql)] = res
        return res
    
    def update_cache(self):
        print("Updating cache to {}...".format(self.cache_fn))
        with open(self.cache_fn, 'w', encoding='utf-8') as fout:
            json.dump(self.cache, fout, ensure_ascii=False, indent=4)
        print("Cache successfully wrote back.")

    def fquery(self, db_name, format, sparql, filename, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=query&username=" + self.username + "&password=" + self.password + "&db_name=" + db_name + "&format=" + format + "&sparql=" + sparql
            self.fGet(strUrl, filename)
        elif request_type == 'POST':        
            strUrl = "/query"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\", \"db_name\": \"' + db_name + '\", \"format\": \"' + format + '\", \"sparql\": \"' + sparql + '\"}'
            self.fPost(strUrl, strPost, filename)
        return

    def drop(self, db_name, is_backup, request_type='GET'):
        if request_type == 'GET':      
            if is_backup:  
                strUrl = "/?operation=drop&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password + "&is_backup=true"
            else:  
                strUrl = "/?operation=drop&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password + "&is_backup=false"
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/drop"
            if is_backup: 
                strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\", \"is_backup\": \"true\"}'
            else: 
                strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\", \"is_backup\": \"false\"}'
            res = self.Post(strUrl, strPost)
        return res

    def monitor(self, db_name, request_type='GET'):    
        if request_type == 'GET':        
            strUrl = "/?operation=monitor&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/monitor"
            strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def checkpoint(self, db_name, request_type='GET'):    
        if request_type == 'GET':        
            strUrl = "/?operation=checkpoint&db_name=" + db_name + "&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/checkpoint"
            strPost = '{\"db_name\": \"' + db_name + '\", \"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def show(self, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=show&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/show"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def getCoreVersion(self, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=getCoreVersion&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/getCoreVersion"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res

    def getAPIVersion(self, request_type='GET'):
        if request_type == 'GET':        
            strUrl = "/?operation=getAPIVersion&username=" + self.username + "&password=" + self.password
            res = self.Get(strUrl)
        elif request_type == 'POST':        
            strUrl = "/getAPIVersion"
            strPost = '{\"username\": \"' + self.username + '\", \"password\": \"' + self.password + '\"}'
            res = self.Post(strUrl, strPost)
        return res
