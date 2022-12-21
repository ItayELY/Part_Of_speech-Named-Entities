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


reg_words = []
ner_to_vec = {}
entities = {}

with open("./data/train") as f:
    line = f.readline()
    c = 0
    while line != '' and c < 200:

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
            if w_list[1] != 'O' and w_list[0] not in entities.keys():
                word_token_index = (inputs.input_ids == tokenizer.encode(w_list[0])[1])[0].nonzero(as_tuple=True)[0][0]
                ner_to_vec[(w_list[0], w_list[1])] = outputs[word_token_index]
                entities[w_list[0].lower()] = w_list[1]
            else:
                if w_list[0] not in reg_words:
                    reg_words.append(w_list[0].lower())
        line = f.readline()

true = 0
true_clasify = 0
total = 0
total_positive = 0




print("started:")
with open("./data/dev_better_pred", 'w') as fw:
    with open("./data/dev_better") as f:

        line = f.readline()
        c = 0
        while line != '':
            new_line = []
            sent_list = line.split(' ')
            split_again = [w.split('/') for w in sent_list]
            sentence = (' ').join([i[0] for i in split_again])

            for w in sent_list:
                w_list = w.split('/')

                if '\n' in w_list[-1]:
                    w_list[-1] = (w_list[-1])[:(len(w_list[-1]) - 1)]
                word2 = '/'.join(w_list[:-1])
                word = w_list[0]
                inputs = tokenizer(sentence, return_tensors="pt")
                outputs = model(**inputs).last_hidden_state[0]
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
                        items = list(ner_to_vec.items())
                        vectors = torch.tensor([items[0][1].tolist()])
                        for i in range(1, len(items)):
                            vectors = torch.cat((vectors, torch.tensor([items[i][1].tolist()])), 0)
                        new_vectors = torch.tensor([vectors.tolist()])

                        new_vec = torch.tensor([vec.tolist()])
                        distances = torch.cdist(new_vec, new_vectors, p=2)
                        pred_pos = 'O'
                        if torch.min(distances) < 7:
                            min_index = torch.argmin(distances)
                            closest_entity = items[min_index][0]

                            pred_pos = closest_entity[1]

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
