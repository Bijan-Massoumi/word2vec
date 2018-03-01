import scipy.sparse as linear
import scipy.sparse.linalg as eigen
import numpy as np
import sys
from preprocess import *
import csv
import itertools
import pdb

unique_word_counter = 0
num_pairs = 0

def collect_sentences(filename):
    all_sentences = 1000000*[None]
    with open(filename,'rb') as f:
        content_iter = csv.reader(f)
        top_index = 0
        for line in content_iter:
            all_sentences[top_index] = preprocess(line[3])
            all_sentences[top_index+1]= preprocess(line[4])
            top_index += 2
        return all_sentences[:top_index]

def compute_ppmi( list_of_sentences ):
    global unique_word_counter
    global num_pairs
    num_words = 0
    distict_word_dict = dict()
    times_pair_dict = dict()
    times_word_occurs = dict()

    for line in list_of_sentences:
        line = line.split(" ")
        sentence_len = len(line)
        num_words += sentence_len
        update_counter_dicts(distict_word_dict,times_word_occurs,line)
        for i in range(0,sentence_len-1):
            if i+3 < sentence_len:
                update_pair_dict(distict_word_dict,times_pair_dict,line[i:i+3])
            else:
                update_pair_dict(distict_word_dict,times_pair_dict,line[i:sentence_len])

    print "finished loading dicts, creating insertion arrays"
    indices = times_pair_dict.keys()
    data = np.ones(len(indices))
    row = np.ones(len(indices), dtype='uint32')
    col = np.ones(len(indices), dtype='uint32')
    for i,pair in enumerate(indices):
        data[i] = max(0, np.log((times_pair_dict[pair]*num_pairs)/ \
                  float((times_word_occurs[pair[0]] * \
                  times_word_occurs[pair[1]]))))
        row[i] = pair[0]
        col[i] = pair[1]
    indices = np.array(indices)
    A = linear.csr_matrix( (data,(row,col)), shape = (unique_word_counter+1,unique_word_counter+1))
    return distict_word_dict , A

def update_counter_dicts(distict_word_dict,times_word_occurs,line):
    global unique_word_counter
    for word in line:
        if word not in distict_word_dict:
            distict_word_dict[word] = unique_word_counter
            unique_word_counter += 1

        if distict_word_dict[word] not in times_word_occurs:
            times_word_occurs[distict_word_dict[word]] = 1
        else:
            times_word_occurs[distict_word_dict[word]] += 1

def update_pair_dict(distict_word_dict,times_pair_dict,window):
    global unique_word_counter
    global num_pairs
    for pair in get_pairs(window,distict_word_dict):
        num_pairs += 1
        if pair in times_pair_dict:
            times_pair_dict[pair] +=1
        else:
            times_pair_dict[pair] = 1


def get_pairs(window,distict_word_dict):
    for i in range(1,len(window)):
        tup = (distict_word_dict[window[0]],\
                distict_word_dict[window[i]])
        yield tup
        yield tup[::-1]

if __name__ == "__main__":
    filename = sys.argv[1]
    sentences = collect_sentences(filename)
    print "finished collecting sentences"
    (word_to_id, matrix) = compute_ppmi(sentences)
    (vals,vectors) = eigen.eigs(matrix,25)

    print matrix.shape[0]
    print linear.csr_matrix(vectors).shape
