'''
Coursework Assignment 3
ANLP 2016, Univeristy of Edinburgh
Author: Stephen Graham (s1601939), Ziwei Wang (s1600429)
Date: 2016-11-26
Some code was provided for us as framework material

Portions of this work adapted from other work released under 
Creative Commons as described here and marked with #### code quote #### 
markers in-line.

  Author: Luke Shrimpton, Sharon Goldwater, Henry Thompson
  Date: 2014-11-01, 2016-11-08
  Copyright: This work is licensed under a Creative Commons
  Attribution-NonCommercial 4.0 International License
  (http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
  redistribute, or modify this work for non-commercial purposes provided
  you retain attribution to any previous author(s).
'''
from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  return log(((N * c_xy) / (c_x * c_y)), 2) 

def PMI_alpha(c_xy, c_x, c_y, N, alpha=0.75):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  numer = c_xy/N
  denom = c_x/N * (c_y**alpha/N**alpha)
  return log(numer/denom, 2)


#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print "Warning: PMI is incorrectly defined"
else:
    print "PMI check passed"

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  #You will need to replace with the real function
  keys = v0.viewkeys() & v1.viewkeys()
  numerator = sum(v0[k] * v1[k] for k in keys)
  v0mag = sqrt(sum(v0[k]**2 for k in v0.viewkeys()))
  v1mag = sqrt(sum(v1[k]**2 for k in v1.viewkeys()))
  return np.divide(numerator, np.multiply(v0mag, v1mag))

def jaccard_similarity(d0,d1):
  '''Compute the similarity between two words by Jaccard method
  heavily inspired by 
  http://dataconomy.com/implementing-the-five-most-popular-similarity-measures-in-python/
  :type d0: dict
  :type d1: dict
  :param d0: co_occurence dictionary for word0
  :param d1: co_occurence dictionary for word1

  :rtype: float
  :return: jaccard similarity ratio

  The idea is to compute the similarity of the two words in a set-wise manner
  by taking the cardinality of the intersection (words that occur in both sets)
  and dividing by the cardinality of the union (all the unique words in each set)
  '''
  set0=set(d0.keys())
  set1=set(d1.keys())
  intersection_cardinality = len(set.intersection(*[set0,set1]))
  union_cardinality = len(set.union(*[set0,set1]))
  return intersection_cardinality/float(union_cardinality)

def dice_coefficient(d0, d1):
  """dice coefficient 2nt/na + nb."""
  if not len(d0) or not len(d1): return 0.0
  a_bigrams = set(d0.keys())
  b_bigrams = set(d1.keys())
  overlap = len(a_bigrams & b_bigrams)
  return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
  '''Creates context vectors for all words, using PPMI.
  These should be sparse vectors.

  :type wids: list of int
  :type o_counts: dict
  :type co_counts: dict of dict
  :type tot_count: int
  :param wids: the ids of the words to make vectors for
  :param o_counts: the counts of each word (indexed by id)
  :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
  :param tot_count: the total number of observations
  :rtype: dict
  :return: the context vectors, indexed by word id
 
  PMI(co_counts[targetid][posid], o_counts[targetid], o_counts[posid], N)
  '''
  vectors = {}
  for wid0 in wids:
      # This works with raw counts, but PMI is probably better.
      # vectors[wid0] = co_counts[wid0]
      ## PMI(co_counts[targetid][posid], o_counts[targetid], o_counts[posid], tot_count)
      ## Dict comprehesion: { k:v for k, v in hand.items() if v }
      vectors[wid0] = {k:v for k,v in ((posid,max(PMI( co_counts[wid0][posid], o_counts[wid0], o_counts[posid], tot_count),0))
             for posid in co_counts[wid0].keys())}
  return vectors

def create_ppmi_vectors_alpha(wids,o_counts, co_counts, tot_count, alpha=0.75):
  '''Creates context vectors for all words, using PPMI-alpha.
  These should be sparse vectors.

  :type wids: list of int
  :type o_counts: dict
  :type co_counts: dict of dict
  :type tot_count: int
  :param wids: the ids of the words to make vectors for
  :param o_counts: the counts of each word (indexed by id)
  :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
  :param tot_count: the total number of observations
  :rtype: dict
  :return: the context vectors, indexed by word id
 
  PMI(co_counts[targetid][posid], o_counts[targetid], o_counts[posid], N)
  '''
  vectors = {}
  for wid0 in wids:
      # This works with raw counts, but PMI is probably better.
      # vectors[wid0] = co_counts[wid0]
      ## PMI(co_counts[targetid][posid], o_counts[targetid], o_counts[posid], tot_count)
      ## Dict comprehesion: { k:v for k, v in hand.items() if v }
      vectors[wid0] = {k:v for k,v in ((posid,max(PMI_alpha( co_counts[wid0][posid], o_counts[wid0], o_counts[posid], tot_count, alpha),0))
             for posid in co_counts[wid0].keys())}
  return vectors


def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(fp.next())
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print "%0.2f\t%-30s\t%d\t%d" % (similarities[pair],word_pair,o_counts[pair[0]],o_counts[pair[1]])

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]



test_words = ["cat", "dog", "mouse", "computer","@justinbieber","#egypt"]
similar_words = ["computer","machine","mac","pc","mouse","phone"]
stemmed_words = [tw_stemmer(w) for w in test_words]

print("Getting all the word id's as a set")
try:
  # At this point we need word2wid
  all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
except NameError:
  # but if we forgot to run the loader in ipython first...
  # Load the data
  # From code by Luke Shrimpton, Sharon Goldwater, Henry Thompson
  #### begin code quote ####
  fp = open("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/wid_word");
  wid2word={}
  word2wid={}
  for line in fp:
    widstr,word=line.rstrip().split("\t")
    wid=int(widstr)
    wid2word[wid]=word
    word2wid[word]=wid
  #### end code quote ####
  all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
print("Creating word vectors...")
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
vectors_alpha = create_ppmi_vectors_alpha(all_wids, o_counts, co_counts, N)

####################################
# demonstrate with cosine similarity
####################################
# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print "Sort by cosine similarity"
print_sorted_pairs(c_sims, o_counts)


c_sims_alpha = {(wid0,wid1): cos_sim(vectors_alpha[wid0],vectors_alpha[wid1]) for (wid0,wid1) in wid_pairs}
print "Sort by cosine similarity (ppmi_alpha)"
print_sorted_pairs(c_sims_alpha, o_counts)


#####################################
# demonstrate with jaccard similarity
#####################################
# compute jaccard similarities for all pairs we consider
j_sims = {(wid0,wid1): jaccard_similarity(vectors[wid0],vectors[wid1])
          for (wid0,wid1) in wid_pairs}

print "Sort by Jaccard similarity"
print_sorted_pairs(j_sims, o_counts)

###################################
# demonstrate with dice_coefficient
###################################
# compute dice_coefficient for all pairs we consider (based on unigram word sets)
d_sims = {(wid0,wid1): dice_coefficient(co_counts[wid0],co_counts[wid1])
          for (wid0,wid1) in wid_pairs}

print "Sort by dice_coefficient"
print_sorted_pairs(d_sims, o_counts)

##------------------------ BEGIN INVESTIGATIONS ----------------------------##
# Define our words to evaluate
print("choosing new words with suspected similarity")
similar_words = ["computer","machine","mac","pc","mouse","phone"]
stemmed_words = [tw_stemmer(w) for w in similar_words]
similar_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
print("making the pairs")
similar_wid_pairs = make_pairs(similar_wids)
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", similar_wids)
print("Creating word vectors...")
similar_vectors = create_ppmi_vectors(similar_wids, o_counts, co_counts, N)
similar_vectors_alpha = create_ppmi_vectors_alpha(similar_wids, o_counts, co_counts, N)

# run each of the methods
c_sims = {(wid0,wid1): cos_sim(similar_vectors[wid0],similar_vectors[wid1]) 
          for (wid0,wid1) in similar_wid_pairs}
c_sims_alpha =  {(wid0,wid1): cos_sim(similar_vectors_alpha[wid0],similar_vectors_alpha[wid1]) 
          for (wid0,wid1) in similar_wid_pairs}
j_sims = {(wid0,wid1): jaccard_similarity(similar_vectors[wid0],similar_vectors[wid1])
          for (wid0,wid1) in similar_wid_pairs}
d_sims = {(wid0,wid1): dice_coefficient(similar_vectors[wid0],similar_vectors[wid1])
          for (wid0,wid1) in similar_wid_pairs}

# save raw data results
print "Sort by cosine similarity"
print_sorted_pairs(c_sims, o_counts)
print "Sort by cosine similarity (ppmi_alpha)"
print_sorted_pairs(c_sims_alpha, o_counts)
print "Sort by Jaccard similarity"
print_sorted_pairs(j_sims, o_counts)
print "Sort by dice_coefficient"
print_sorted_pairs(d_sims, o_counts)

# graph the results (and save to file)


##------------------------ BEGIN INVESTIGATIONS ----------------------------##
# Define our words to evaluate
print("choosing new words with suspected difference")
different_words = ["cat", "shovel", "car", "#egypt", "trump", "easy"]
stemmed_words = [tw_stemmer(w) for w in different_words]
different_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
print("making the pairs")
similar_wid_pairs = make_pairs(different_wids)
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", different_wids)
print("Creating word vectors...")
different_vectors = create_ppmi_vectors(different_wids, o_counts, co_counts, N)
different_vectors_alpha = create_ppmi_vectors_alpha(different_wids, o_counts, co_counts, N)

# run each of the methods
c_sims = {(wid0,wid1): cos_sim(different_vectors[wid0],different_vectors[wid1]) 
          for (wid0,wid1) in similar_wid_pairs}
c_sims_alpha =  {(wid0,wid1): cos_sim(different_vectors_alpha[wid0],different_vectors_alpha[wid1]) 
          for (wid0,wid1) in similar_wid_pairs}
j_sims = {(wid0,wid1): jaccard_similarity(different_vectors[wid0],different_vectors[wid1])
          for (wid0,wid1) in similar_wid_pairs}
d_sims = {(wid0,wid1): dice_coefficient(different_vectors[wid0],different_vectors[wid1])
          for (wid0,wid1) in similar_wid_pairs}

# save raw data results
print "Sort by cosine similarity"
print_sorted_pairs(c_sims, o_counts)
print "Sort by cosine similarity (ppmi_alpha)"
print_sorted_pairs(c_sims_alpha, o_counts)
print "Sort by Jaccard similarity"
print_sorted_pairs(j_sims, o_counts)
print "Sort by dice_coefficient"
print_sorted_pairs(d_sims, o_counts)

# graph the results (and save to file)


def get_similarity(wlist,functionlist=[cos_sim,jaccard_similarity,dice_coefficient]):
  stemmed_words = [tw_stemmer(w) for w in wlist]
  different_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
  print("making the pairs")
  similar_wid_pairs = make_pairs(different_wids)
  (o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", different_wids)
  print("Creating word vectors...")
  different_vectors = create_ppmi_vectors(different_wids, o_counts, co_counts, N)
  results = []
  for method in functionlist:
    results.append(('{}'.format(method),{(wid0,wid1): method(different_vectors[wid0],different_vectors[wid1]) 
          for (wid0,wid1) in similar_wid_pairs}))
  return results
