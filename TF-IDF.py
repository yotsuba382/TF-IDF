# coding: UTF-8
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

argvs = sys.argv
readFile = argvs[1]

np.set_printoptions(precision=2)

#docs = np.loadtxt(readFile, delimiter=" ")
docs = np.array(["The Argentine\'s consistency in hitting the net is as astonishing due to the number of goals he has scored as it is due to the time in which he scored them. Barça's number 10 does just not tire of scoring, of winning of providing assists or of playing with pure joy. And as long as he keeps doing all that, we'll have Messi around for still a while to come. Messi's goal in the Copa del Rey final against Sevilla took the boy from Rosario on to 40 goals this ampaign, a tally he has sustained since 2009-10, Guardiola's second in the Barça dugout. And with five games remaining in the league (against Deportivo, Real Madrid, Villarrea, Levante and Real Sociedad), the Barça striker could yet increase his goal tally and beat the 41 goals he scored in 2013-14 and 2015-16. His record was in 2011-12 with a barnstorming 73 strikes in 60 games."])

print(docs)
 
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
 
print(vecs.toarray())