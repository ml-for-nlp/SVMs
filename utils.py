import random
import numpy as np
from itertools import product
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def get_data(lst):
#get a list of filepaths & serve a dictionary class:vector
    out = {}
    for f in lst:
        d = parse_file(f)
        out[f] = d
    return out

def format_cls(lst):
#serve str with list of available cls
    out = ''
    for item in lst:
        out += item + ', '
    return out[:-2] + '.'


def get_train_size(cl, dic):
#parse and check user input
    tot = len(dic)

    s = '{} has {} docs. '.format(cl, str(tot)) + \
        'How many for training? '
    err = 'Too many. Please enter a lower number.\n'

    firsttime = True
    while True:
        n = int(input(s if firsttime else err + s))
        firsttime = False
        if n < tot:
            return n

def make_arrays(space, n):
#serve numpy arrays and source docs for SVM
    training = []
    training_docs = []
    test = []
    test_docs = []
    c = 0

    keys=list(space.keys())
    random.shuffle(keys)

    for key in keys:
        value=space[key]
        if c < n:
            training.append(value)
            training_docs.append(key)
            c += 1
        else:
            test.append(value)
            test_docs.append(key)
    return np.array(training), np.array(test), training_docs, test_docs

def make_labels(size1, size2):
    out = []
    for i in range(size1):
        out.append(1)
    for i in range(size2):
        out.append(2)
    return np.array(out)

def normalize(vec):
#calc normalized vector
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec 
    return vec / norm

def get_queries(f):
#create dict of lemma:vector
    out = {}
    with open(f, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line  = line.rstrip().split('::')
        #line  = line.rstrip().split(' ')
        lemma = line[0]
        vec   = normalize(np.array([float(i) \
            for i in line[1].split()]))
        #vec   = normalize(np.array([int(i) for i in lines[1:]]))
        out[lemma] = vec

    return out

def parse_file(filename):
#get data from csv file
    print("Parsing",filename,"...")
    dm = {}
    with open(filename) as f:
        for line in f:
            try:
                fields = line.rstrip('\n').split(',')
                doc = fields[0]
                vector = np.array([float(i) for i in fields[1:]])
                dm[doc] = vector
            except:
                pass
    return dm

