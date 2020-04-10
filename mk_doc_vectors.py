#USAGE: python3 mk_doc_vectors.py

import re
import sys
import numpy as np
from math import isnan
from os import listdir
from os.path import isfile, isdir, join
from sklearn.decomposition import PCA

feature_type = sys.argv[1]    

def read_vocab():
    with open('./data/vocab_file.txt','r') as f:
        vocab = f.read().splitlines()
    return vocab

def get_words(l):
    l=l.lower()
    words = {}
    for word in l.split():
        if word in words:
            words[word]+=1
        else:
            words[word]=1
    return words

def get_ngrams(l,n):
    l = l.lower()
    ngrams = {}
    for i in range(0,len(l)-n+1):
        ngram = l[i:i+n]
        if ngram in ngrams:
            ngrams[ngram]+=1
        else:
            ngrams[ngram]=1
    return ngrams

def normalise(v):
    return v / sum(v)

def run_PCA(d,docs):
    m = []
    retained_docs = []
    for url in docs:
        if not isnan(sum(d[url])) and sum(d[url]) != 0:
            m.append(d[url])
            retained_docs.append(url)
    pca = PCA(n_components=300)
    pca.fit(m)
    m_300d = pca.transform(m)
    return np.array(m_300d), retained_docs


def clean_docs(d,docs):
    m = []
    retained_docs = []
    for url in docs:
        if not isnan(sum(d[url])) and sum(d[url]) != 0:
            m.append(d[url])
            retained_docs.append(url)
    return np.array(m), retained_docs



d = './data'
catdirs = [join(d,o) for o in listdir(d) if isdir(join(d,o))]
vocab = read_vocab()

for cat in catdirs:
    print(cat)
    url = ""
    docs = []
    vecs = {}
    doc_file = open(join(cat,"linear.txt"),'r')
    for l in doc_file:
        l=l.rstrip('\n')
        if l[:4] == "<doc":
            m = re.search("date=(.*)>",l)
            url = m.group(1).replace(',',' ')
            docs.append(url)
            vecs[url] = np.zeros(len(vocab))
        if l[:5] == "</doc":
            vecs[url] = normalise(vecs[url])
            print(url,sum(vecs[url]))
        if feature_type == "ngrams":
            for i in range(3,7):
                ngrams = get_ngrams(l,i)
                for k,v in ngrams.items():
                    if k in vocab:
                        vecs[url][vocab.index(k)]+=v
        if feature_type == "words":
            words = get_words(l)
            for k,v in words.items():
                if k in vocab:
                    vecs[url][vocab.index(k)]+=v
                 
        
    
    doc_file.close()
    m,retained_docs = clean_docs(vecs,docs)
    print("------------------")
    print("NUM ORIGINAL DOCS:", len(docs))
    print("NUM RETAINED DOCS:", len(retained_docs))
 
    vec_file = open(join(cat,"vecs.csv"),'w')
    for i in range(len(retained_docs)):
            vec_file.write(retained_docs[i]+','+','.join([str(v) for v in m[i]])+'\n') 
    vec_file.close()
