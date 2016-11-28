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
import pandas as pd


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

def X_get_plot_data(similarities, o_counts, first=0, last=100):
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
  :return: list of word pair similarity data for plotting
  '''
  plot_data = []
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    plot_data.append([similarities[pair],word_pair,o_counts[pair[0]],o_counts[pair[1]]])
  return plot_data

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]

def test_sims(word_set, description='*DEFAULT TEST NAME*', stemmed=True):
  '''Wrapper function to generate our set of similarity tests for a given set of words

  :type word_set: list
  :param word_set: list of words for generating distributional similarity indices
  :return type: pandas Dataframe
  :return: the aggregated results test as dataframe keyed on word pairs
  '''
  print('Generating test: {}'.format(description))

  if stemmed: stemmed_words = [tw_stemmer(w) for w in word_set]  # if stemming is requested (default)
  unique_wids = set([word2wid[x] for x in stemmed_words])        # remove duplicates (if any)
  wid_pairs = make_pairs(unique_wids)                            # create pairs 
  
  # generate the counts
  (o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", unique_wids)


  print("Creating word vectors...")
  vectors = create_ppmi_vectors(unique_wids, o_counts, co_counts, N)
  vectors_alpha = create_ppmi_vectors_alpha(unique_wids, o_counts, co_counts, N)

  # run each of the methods in our test space
  c_sims       = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
  c_sims_alpha = {(wid0,wid1): cos_sim(vectors_alpha[wid0],vectors_alpha[wid1]) for (wid0,wid1) in wid_pairs}
  j_sims       = {(wid0,wid1): jaccard_similarity(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
  d_sims       = {(wid0,wid1): dice_coefficient(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

  # display raw data results
  print "Sort by cosine similarity"
  print_sorted_pairs(c_sims, o_counts)
  print "Sort by cosine similarity (ppmi_alpha)"
  print_sorted_pairs(c_sims_alpha, o_counts)
  print "Sort by Jaccard similarity"
  print_sorted_pairs(j_sims, o_counts)
  print "Sort by dice_coefficient"
  print_sorted_pairs(d_sims, o_counts)

  # aggregate the results
  # get all the keys from all the tests (because they might differ...)
  result_keys = set(c_sims.keys() + c_sims_alpha.keys() + j_sims.keys() + d_sims.keys())
  result_frame = pd.DataFrame(index = result_keys, 
                              columns = ['word_pairs','c_sims','c_sims_alpha','j_sims','d_sims'])

  # populate the frame
  for k in result_keys: result_frame.set_value(k, 'word_pairs', (wid2word[k[0]], wid2word[k[1]]))
  for k in c_sims.keys(): result_frame.set_value(k, 'c_sims', c_sims[k])
  for k in c_sims_alpha.keys(): result_frame.set_value(k, 'c_sims_alpha', c_sims_alpha[k])
  for k in j_sims.keys(): result_frame.set_value(k, 'j_sims', j_sims[k])
  for k in d_sims.keys(): result_frame.set_value(k, 'd_sims', d_sims[k])

  return result_frame
  #-----------end test generator function -----------------------#

def plot_results(results,legend,title='*SET TITLE*',figname=None):
  ''' Plot a set of results on a single plot for a set of results
  :type results: pandas DataFrame
  :type legend: list
  :type title: string
  :type figname: string
  :param results: dataframe indexed on word pairs with a column for each set of results
  :param legend: List of friendly names of the tests in the results
         e.g. ['Cosine (PPMI)','Cosine (PPMI-alpha)']
  :param title: A string containing the figure title
  :param figname: if not None, save the generated figure as a file named figname
  :return None
  '''

  # Build an appropriate figure
  fig = plt.figure()
  ax = fig.add_subplot(111)

  xlabels = results['word_pairs']
  xrange = range(len(xlabels))
  for c in [col for col in results.columns.values if col != 'word_pairs']:
    ax.plot(xrange,results[c])
    ax.set_xticklabels(xlabels,rotation=60)
  plt.legend(legend)
  plt.title(title)
  plt.tight_layout()
  plt.show()
  if figname is not None: fig.savefig(figname)


#--------------------------------------------------------------------#
# begin the preliminary tests
#--------------------------------------------------------------------#
test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
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

test_names = ['cos(PPMI)','cos(PPMI-alpha)','Jaccard','Dice']
print('Beginning test series for these measures: {}'.format(test_names))
print('-' * 40)
# test 1
test_positive = test_sims(["computer","machine","mac","pc","mouse","phone"],'Manually selected similar words')
plot_results(test_positive.sort_values(['c_sims'],ascending=False), 
             test_names, 
             'Manually selected word pairs anticipated to be similar\nSorted by cos(PPMI)',
             'test_pos_cos.svg')

#test 2
test_negative = test_sims(["cat", "shovel", "car", "#egypt", "trump", "easy"],'Manually selected different words')
plot_results(test_negative.sort_values(['c_sims'],ascending=False), 
             test_names, 
             'Manually selected word pairs anticipated to be dissimilar\nSorted by cos(PPMI)',
             'test_neg_cos.svg')
print('=' * 40)
print('End of tests.')

##------------------------- END INVESTIGATIONS -----------------------------##
