import random
import numpy as np
from itertools import product
from math import sqrt
from matplotlib import cm
import matplotlib.pyplot as plt
from pandas import DataFrame
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
#serve numpy arrays and source urls for SVM
    training = []
    training_urls = []
    test = []
    test_urls = []
    c = 0

    keys=list(space.keys())
    random.shuffle(keys)

    for key in keys:
        value=space[key]
        if c < n:
            training.append(value)
            training_urls.append(key)
            c += 1
        else:
            test.append(value)
            test_urls.append(key)
    return np.array(training), np.array(test), training_urls, test_urls

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
                url = fields[0]
                vector = np.array([float(i) for i in fields[1:]])
                dm[url] = vector
            except:
                pass
    return dm

def plot_confmat(cm, classes, normalized, \
    title = 'Confusion matrix', cmap = plt.cm.Blues):

    if normalized:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix:')
    else:
        print('Confusion matrix, without normalization:')
    print(cm)
    print()

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalized else 'd'
    threshold = cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), \
        horizontalalignment = 'center', \
                color = 'white' if cm[i, j] > threshold else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_confmat(y_test, y_pred, t1, t2):
    #compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision = 2)

    #plot non-normalized confusion matrix
    plt.figure()
    plot_confmat(cm, classes = [t1,t2], normalized = False, \
        title = 'Confusion matrix, without normalization')
    plt.savefig("confusion.png")

    #plot normalized confusion matrix
    plt.figure()
    plot_confmat(cm, classes = [t1,t2], normalized = True, \
        title = 'Normalized confusion matrix')
    plt.savefig("confusion-norm.png")
