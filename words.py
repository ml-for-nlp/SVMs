#USAGE: python3 words.py <word size>
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

for cat in catdirs:
    words = {}
    f = open(join(cat,'linear.txt'),'r')
    for l in f:
        l = l.rstrip('\n').lower()
        for word in l.split():
            if contain_symbols(word):
                continue
            if word in words:
                words[word]+=1
            else:
                words[word]=1
    f.close()

    wordfile = open(join(cat,"linear.words"),'w')
    for k in sorted(words, key=words.get, reverse=True):
        wordfile.write(k+'\t'+str(words[k])+'\n')
    wordfile.close() 
