# coding: UTF-8
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys,csv

def readFile():

    docs = []
    np.set_printoptions(precision=2)
    
    argvs = sys.argv
    file = argvs[1]
    f = open(file,"r")
    
    first = f.readlines()
    for line in first:
        docs.append(line)
    f.close()

    #print(docs)

    return docs

def makeCSV(X):
    f = open('data.csv', 'w')

    writer = csv.writer(f, lineterminator='\n')
    #writer.writerow(list)
    writer.writerows(X)

    f.close()

if __name__ == '__main__':

    docs = readFile()

    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)

    X = vecs.toarray()
    print(vecs.toarray())

    #make Txt file
    np.savetxt('input_data.txt', X,fmt="%.4f")

    makeCSV(X)