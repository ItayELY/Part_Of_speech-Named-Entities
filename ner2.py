from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import torch
import gensim.downloader
from numpy.linalg import norm
import numpy as numpy
import torch
from scipy.spatial import distance
#'O', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC', 'B-PER', 'B-MISC', 'B-ORG', 'B-LOC']

#glove_vectors = gensim.downloader.load( 'glove-wiki-gigaword-100')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = RobertaModel.from_pretrained('roberta-base')
model2 = RobertaForMaskedLM.from_pretrained("roberta-base")


reg_words = []

ner_to_vec2 = {}
entities = {}
avg_vectors = {}
ners_counts = {}
with open("./data/train") as f:
    line = f.readline()
    c = 0
    while line != '' and c < 500:

        if c % 10 == 0:
            print(c)
        c += 1
        sent_list = line.split(' ')

        split_again = [w.split('/') for w in sent_list]
        sentence = (' ').join([i[0] for i in split_again])
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs).last_hidden_state[0]
        for w in sent_list:
            w_list = w.split('/')
            if '\n' in w_list[1]:
                w_list[1] = (w_list[1])[:(len(w_list[1]) - 1)]
            if w_list[0] not in entities.keys() and w_list[1] in ['I-PER', 'I-MISC', 'I-ORG', 'I-LOC',
                                   'B-PER', 'B-MISC', 'B-ORG', 'B-LOC']:
                if w_list[1] not in ners_counts.keys():
                    ners_counts[w_list[1]] = 1
                #if ners_counts[w_list[1]] >= 80:
                 #   continue
                ners_counts[w_list[1]] += 1
                word_token_index = (inputs.input_ids == tokenizer.encode(w_list[0])[1])[0].nonzero(as_tuple=True)[0][0]
                vector = outputs[word_token_index]

                try:
                    ner_to_vec2[w_list[1]] = torch.cat((ner_to_vec2[w_list[1]], torch.tensor([vector.tolist()])), 0)
                except Exception as e:
                    print(e)
                    ner_to_vec2[w_list[1]] = torch.tensor([vector.tolist()])
                entities[w_list[0].lower()] = w_list[1]

            else:
                if w_list[0] not in reg_words:
                    reg_words.append(w_list[0].lower())
        line = f.readline()
for k in avg_vectors.keys():
    avg_vectors[k] = torch.div(avg_vectors[k], ners_counts[k])

true = 0
true_clasify = 0
total = 0
total_positive = 0




print("started:")
'''with open("./data/dev_pred", 'w') as fw:
    with open("./data/dev") as f:

        line = f.readline()
        c = 0
        while line != '':
            new_line = []
            sent_list = line.split(' ')
            split_again = [w.split('/') for w in sent_list]
            sentence = (' ').join([i[0] for i in split_again])
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs).last_hidden_state[0]
            for w in sent_list:
                w_list = w.split('/')

                if '\n' in w_list[-1]:
                    w_list[-1] = (w_list[-1])[:(len(w_list[-1]) - 1)]
                word2 = '/'.join(w_list[:-1])
                word = w_list[0]
                pred_pos = ''
                if word.lower() not in reg_words:
                    if word.lower() in entities.keys():
                        real_pos = w_list[-1]
                        if real_pos == entities[word.lower()]:
                            true += 1
                            true_clasify += 1
                        total += 1
                        if real_pos != 'O':
                            total_positive += 1
                        pred_pos = entities[word.lower()]

                    else:
                        word_token_index = \
                            ((inputs.input_ids == tokenizer.encode(word)[1])[0].nonzero(as_tuple=True))[0][0]
                        vec = outputs[word_token_index]
                        #items = list(ner_to_vec.items())
                        softmax = {}
                        new_vec = torch.tensor([vec.tolist()])
                        min = 20
                        min_nes = ''
                        dists = {}
                        for item in ner_to_vec2.items():
                            distances = torch.cdist(new_vec, item[1], p=2)[0]
                            for d in distances:
                                dists[d] = item[0]
                            if min > torch.min(distances):
                                min = torch.min(distances)
                                min_nes = item[0]
                        i = 0
                        occurances = {}
                        for k in sorted(dists.keys()):
                            if i == 5:
                                break
                            try:
                                occurances[dists[k]] += 1 + ((1/ners_counts[dists[k]]) * 20)
                            except:
                                occurances[dists[k]] = 1 + ((1/ners_counts[dists[k]]) * 20)
                            i += 1
                            #softmax[item[0]] = torch.div(torch.sum(distances), distances.size(dim=1)) + 1*(1/distances.size(dim=1))
                        pred_pos = 'O'
                       # softmax["B-MISC"] += 0.5
 #                       softmax["B-PER"] += 0.5
                       # softmax["B-LOC"] += 0.5
 #                       softmax["B-ORG"] += 0.5
                       # softmax["I-ORG"] -= 0.4
                      #  softmax["I-MISC"] -= 0.25
                        if min < 6 or (max([i for i in occurances.values()]) >= 5 and min<7) :
                           # min_index = torch.argmin(distances)
                            #closest_entity = min_nes

                            pred_pos = max(occurances, key=occurances.get)



                        real_pos = w_list[-1]
                        if real_pos == pred_pos:
                            true += 1
                            if real_pos != 'O':
                                true_clasify += 1
                                total_positive += 1
                        total += 1
                        if pred_pos == 'O':
                            reg_words.append(word.lower())
                        else:
                            entities[word.lower()] = pred_pos
                            pass
                else:
                    pred_pos = 'O'
                    real_pos = w_list[-1]
                    if real_pos == pred_pos:
                        true += 1
                    else:
                        total_positive += 1
                    total += 1
                if word == '':
                    word = '/'
                if pred_pos not in['O', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC',
                                   'B-PER', 'B-MISC', 'B-ORG', 'B-LOC']:
                    pred_pos = 'O'
                new_line.append('/'.join([word2, pred_pos]))
                if c % 100 == 0:
                    print(c)
                c += 1
            new_line.append('\n')
            fw.write(' '.join(new_line))
            line = f.readline()
print("percision: " + str(true/total) + " " + str(true) + " recall: " + str(true_clasify/total_positive))
'''







with open("./NER_preds.txt", 'w') as fw:
    with open("./data/test.blind") as f:

        line = f.readline()
        c = 0
        while line != '':
            new_line = []
            sent_list = line.split(' ')

            sentence = line
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs).last_hidden_state[0]
            for w in sent_list:
                if '\n' in w:
                    w = w[:len(w) - 1]
                pred_pos = ''
                if w.lower() not in reg_words:
                    if w.lower() in entities.keys():
                        real_pos = w_list[-1]
                        if real_pos == entities[w.lower()]:
                            true += 1
                            true_clasify += 1
                        total += 1
                        if real_pos != 'O':
                            total_positive += 1
                        pred_pos = entities[w.lower()]

                    else:
                        word_token_index = \
                            ((inputs.input_ids == tokenizer.encode(w)[1])[0].nonzero(as_tuple=True))[0][0]
                        vec = outputs[word_token_index]
                        #items = list(ner_to_vec.items())
                        softmax = {}
                        new_vec = torch.tensor([vec.tolist()])
                        min = 20
                        min_nes = ''
                        dists = {}
                        for item in ner_to_vec2.items():
                            distances = torch.cdist(new_vec, item[1], p=2)[0]
                            for d in distances:
                                dists[d] = item[0]
                            if min > torch.min(distances):
                                min = torch.min(distances)
                                min_nes = item[0]
                        i = 0
                        occurances = {}
                        for k in sorted(dists.keys()):
                            if i == 5:
                                break
                            try:
                                occurances[dists[k]] += 1 + ((1/ners_counts[dists[k]]) * 20)
                            except:
                                occurances[dists[k]] = 1 + ((1/ners_counts[dists[k]]) * 20)
                            i += 1
                            #softmax[item[0]] = torch.div(torch.sum(distances), distances.size(dim=1)) + 1*(1/distances.size(dim=1))
                        pred_pos = 'O'
                       # softmax["B-MISC"] += 0.5
 #                       softmax["B-PER"] += 0.5
                       # softmax["B-LOC"] += 0.5
 #                       softmax["B-ORG"] += 0.5
                       # softmax["I-ORG"] -= 0.4
                      #  softmax["I-MISC"] -= 0.25
                        if min < 6 or (max([i for i in occurances.values()]) >= 5 and min<7) :
                           # min_index = torch.argmin(distances)
                            #closest_entity = min_nes

                            pred_pos = max(occurances, key=occurances.get)



                
                else:
                    pred_pos = 'O'

                if pred_pos not in['O', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC',
                                   'B-PER', 'B-MISC', 'B-ORG', 'B-LOC']:
                    pred_pos = 'O'
                new_line.append('/'.join([w, pred_pos]))
                if c % 100 == 0:
                    print(c)
                c += 1
            new_line.append('\n')
            fw.write(' '.join(new_line))
            line = f.readline()

