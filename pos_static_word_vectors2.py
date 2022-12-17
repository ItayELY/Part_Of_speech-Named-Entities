import gensim.downloader
from numpy.linalg import norm
import numpy as numpy
import torch
from scipy.spatial import distance

glove_vectors = gensim.downloader.load( 'glove-wiki-gigaword-50')



class Word():
    def __init__(self, w):
        self.w = w
        self.counts = {}

words = {}
vectors_to_pos = {}
pos_count = {}
with open("./data/ass1-tagger-train") as f:
    line = f.readline()
    c = 0
    while line != '':
        c += 1
        sent_list = line.split(' ')

        for w in sent_list:

            w_list = w.split('/')
            try:
                vectors_to_pos[glove_vectors[w_list[0].lower()]] = w_list[1]
            except:
                if w_list[0] not in words.keys():
                    new_w = Word(w_list[0])
                    words[w_list[0]] = new_w
                if w_list[1] not in words[w_list[0]].counts:
                    words[w_list[0]].counts[w_list[1]] = 0
                words[w_list[0]].counts[w_list[1]] += 1
            if w_list[1] not in pos_count.keys():
                pos_count[w_list[1]] = 0
            pos_count[w_list[1]] += 1
        line = f.readline()
