from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import torch
import gensim.downloader
from numpy.linalg import norm
import numpy as numpy
import torch
from scipy.spatial import distance

#glove_vectors = gensim.downloader.load( 'glove-wiki-gigaword-100')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaModel.from_pretrained('roberta-base')
model2 = RobertaForMaskedLM.from_pretrained("roberta-base")


class Word():
    def __init__(self, w):
        self.w = w
        self.counts = {}

words = {}
word_to_vec = {}
pos_to_vec = {}
pos_count = {}
with open("./data/ass1-tagger-train") as f:
    line = f.readline()
    c = 0
    while line != '':
        c += 1

        if c % 1000 == 0:
            print(c)
        sent_list = line.split(' ')

        inputs = None
        if c <= 40:
            print(c)
            split_again = [w.split('/') for w in sent_list]
            sentence = (' ').join([i[0] for i in split_again])
            inputs = tokenizer(sentence, return_tensors="pt")
        for w in sent_list:
            w_list = w.split('/')
            if c <= 40:
                word_token_index = (inputs.input_ids == tokenizer.encode(w_list[0])[1])[0].nonzero(as_tuple=True)[0][0]
                pos_to_vec[(w_list[0], w_list[1])] = model(**inputs).last_hidden_state[0][word_token_index]
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

'''
for w in words.keys():
    try:
        word_to_vec[w] = glove_vectors[w.lower()]
    except:
        pass
'''

true = 0
total = 0




print("started:")
'''with open("./data/ass1-tagger-dev") as f:
    line = f.readline()
    c = 0
    while line != '':

        sent_list = line.split(' ')
        split_again = [w.split('/') for w in sent_list]
        sentence = (' ').join([i[0] for i in split_again])
        inputs = tokenizer(sentence, return_tensors="pt")
        for w in sent_list:
            w_list = w.split('/')
            word = w_list[0]

            if word != '.\n':
                if c%1000 == 0:
                    print(c)
                c += 1
                if word not in words.keys():
                    word_token_index = \
                        ((inputs.input_ids == tokenizer.encode(word)[1])[0].nonzero(as_tuple=True))[0][0]
                    vec = model(**inputs).last_hidden_state[0][word_token_index]
                    items = list(pos_to_vec.items())
                    vectors = torch.tensor([items[0][1].tolist()])
                    for i in range(1, len(items)):
                        vectors = torch.cat((vectors, torch.tensor([items[i][1].tolist()])), 0)
                    new_vectors = torch.tensor([vectors.tolist()])

                    new_vec = torch.tensor([vec.tolist()])
                    distances = torch.cdist(new_vec, new_vectors, p=2)[0]
                    min_index = torch.argmin(distances)
                    closest_pair = items[min_index][0]
                    real_pos = w_list[1]
                    pred_pos = closest_pair[1]
                    if real_pos == pred_pos:
                        true += 1
                    total += 1

                    continue

                if word in words.keys():
                    preds = {}
                    counts = words[word].counts
                    real_pos = w_list[1]
                    pred_pos = max(counts, key=counts.get)
                    if real_pos == pred_pos:
                        true += 1
                    total+= 1


        line = f.readline()
print("accuracy: " + str(true/total) + " " + str(true) + " correct out of " + str(total))
'''
with open("POS_preds_3.txt", "w") as f1:
    with open("./data/ass1-tagger-test-input") as f2:
        line = f2.readline()

        while line != '':
            new_line = []
            sent_list = line.split(' ')
            inputs = tokenizer(line, return_tensors="pt")
            for w in sent_list:
                word = w

                if word != '.\n':
                    if word not in words.keys():
                        try:
                            word_token_index = \
                                ((inputs.input_ids == tokenizer.encode(word)[1])[0].nonzero(as_tuple=True))[0][0]
                            vec = model(**inputs).last_hidden_state[0][word_token_index]
                            items = list(pos_to_vec.items())
                            vectors = torch.tensor([items[0][1].tolist()])
                            for i in range(1, len(items)):
                                vectors = torch.cat((vectors, torch.tensor([items[i][1].tolist()])), 0)
                            new_vectors = torch.tensor([vectors.tolist()])

                            new_vec = torch.tensor([vec.tolist()])
                            distances = torch.cdist(new_vec, new_vectors, p=2)[0]
                            min_index = torch.argmin(distances)
                            closest_pair = items[min_index][0]
                            pred_pos = closest_pair[1]
                            new_line.append('/'.join([w, pred_pos]))

                        except Exception as inst:
                            print(inst)
                            pred_pos = max(pos_count, key=pos_count.get)
                            new_line.append('/'.join([w, pred_pos]))
                    else:
                        pred_pos = max(words[word].counts, key=words[word].counts.get)
                        new_line.append('/'.join([w, pred_pos]))
            new_line.append('./.\n')
            f1.write(' '.join(new_line))
            line = f2.readline()