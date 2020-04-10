#USAGE: python3 ngrams.py <ngram size>
import string
import sys
from os import listdir
from os.path import isfile, isdir, join
    
def contain_symbols(s):
    symbols = [c for c in string.punctuation]
    symbols.extend([d for d in string.digits])
    r = any(c in s for c in symbols) 
    return r


d = './data/'
catdirs = [join(d,o) for o in listdir(d) if isdir(join(d,o))]
n = int(sys.argv[1])

for cat in catdirs:
    ngrams = {}
    f = open(join(cat,'linear.txt'),'r')
    for l in f:
        l = l.rstrip('\n').lower()
        for i in range(len(l)-n+1):
            ngram = l[i:i+n]
            if contain_symbols(ngram):
                continue
            if ngram in ngrams:
                ngrams[ngram]+=1
            else:
                ngrams[ngram]=1
    f.close()

    ngramfile = open(join(cat,"linear."+str(n)+".ngrams"),'w')
    for k in sorted(ngrams, key=ngrams.get, reverse=True):
        ngramfile.write(k+'\t'+str(ngrams[k])+'\n')
    ngramfile.close() 
