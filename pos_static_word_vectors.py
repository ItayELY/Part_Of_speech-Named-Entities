import gensim.downloader
from numpy.linalg import norm
import numpy as numpy
import torch
from scipy.spatial import distance

glove_vectors = gensim.downloader.load( 'glove-wiki-gigaword-100')





class Word():
    def __init__(self, w):
        self.w = w
        self.counts = {}

words = {}

pos_count = {}
with open("./data/ass1-tagger-train") as f:
    line = f.readline()
    c = 0
    while line != '':
        c += 1
        sent_list = line.split(' ')
        for w in sent_list:

            w_list = w.split('/')
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

word_to_vec = {}
for w in words.keys():
    try:
        word_to_vec[w] = glove_vectors[w.lower()]
    except:
        pass
true = 0
total = 0

print("started:")
'''with open("./data/ass1-tagger-dev") as f:
    line = f.readline()
    c = 0
    while line != '':

        sent_list = line.split(' ')
        for w in sent_list:
            w_list = w.split('/')
            word = w_list[0]
            if word != '.\n':
                if c%1000 == 0:
                    print(c)
                c += 1
                if word in words.keys():
                    preds = {}
                    counts = words[word].counts
               

                    real_pos = w_list[1]
                    pred_pos = max(counts, key=counts.get)
                    if real_pos == pred_pos:
                        true += 1
                else:
                    try:
                        vec = numpy.array(glove_vectors[word.lower()])

                        closest_word = None
                        closest_num = 0


                                #cos_sim = numpy.dot(vec, v) / (norm(vec) * norm(v))
                                #cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                                #cos_sim = cos(vec, v)
                        items = list(word_to_vec.items())
                        vecs = [i[1] for i in items]
                        distances = distance.cdist([vec], vecs , "cosine")[0]
                        min_index = numpy.argmin(distances)
                        closest_word = items[min_index][0]
                        real_pos = w_list[1]
                        pred_pos = max(words[closest_word].counts, key=words[closest_word].counts.get)
                        if real_pos == pred_pos:
                            true += 1
                    except Exception as inst:
                        real_pos = w_list[1]
                        pred_pos = max(pos_count, key=pos_count.get)
                        if real_pos == pred_pos:
                            true += 1
                total += 1


        line = f.readline()
print("accuracy: " + str(true/total) + " " + str(true) + " correct out of " + str(total))
'''

with open("POS_preds_2.txt", "w") as f1:
    with open("./data/ass1-tagger-test-input") as f2:
        line = f2.readline()

        while line != '':
            new_line = []
            sent_list = line.split(' ')
            for w in sent_list:
                word = w
                if word != '.\n':
                    if word not in words.keys():
                        try:
                            vec = numpy.array(glove_vectors[word.lower()])
                            closest_word = None
                            closest_num = 0

                            items = list(word_to_vec.items())
                            vecs = [i[1] for i in items]
                            distances = distance.cdist([vec], vecs, "cosine")[0]
                            min_index = numpy.argmin(distances)
                            closest_word = items[min_index][0]
                            pred_pos = max(words[closest_word].counts, key=words[closest_word].counts.get)
                            new_line.append('/'.join([w, pred_pos]))
                        except Exception as inst:
                            pred_pos = max(pos_count, key=pos_count.get)
                            new_line.append('/'.join([w, pred_pos]))
                    else:
                        pred_pos = max(words[word].counts, key=words[word].counts.get)
                        new_line.append('/'.join([w, pred_pos]))
            new_line.append('./.\n')
            f1.write(' '.join(new_line))
            line = f2.readline()
