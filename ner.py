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


words = {}
ner_to_vec = {}
with open("./data/train") as f:
    line = f.readline()
    c = 0
    while line != '':
        c += 1
        if c % 1000 == 0:
            print(c)
        sent_list = line.split(' ')

        split_again = [w.split('/') for w in sent_list]
        sentence = (' ').join([i[0] for i in split_again])
        inputs = tokenizer(sentence, return_tensors="pt")
        for w in sent_list:
            w_list = w.split('/')
            if w_list[1] != '/O':
                word_token_index = (inputs.input_ids == tokenizer.encode(w_list[0])[1])[0].nonzero(as_tuple=True)[0][0]
                ner_to_vec[(w_list[0], w_list[1])] = model(**inputs).last_hidden_state[0][word_token_index]
        line = f.readline()

true = 0
total = 0
total_positive = 0




print("started:")
with open("./data/dev") as f:
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
                word_token_index = \
                    ((inputs.input_ids == tokenizer.encode(word)[1])[0].nonzero(as_tuple=True))[0][0]
                vec = model(**inputs).last_hidden_state[0][word_token_index]
                items = list(ner_to_vec.items())
                vectors = torch.tensor([items[0][1].tolist()])
                for i in range(1, len(items)):
                    vectors = torch.cat((vectors, torch.tensor([items[i][1].tolist()])), 0)
                new_vectors = torch.tensor([vectors.tolist()])

                new_vec = torch.tensor([vec.tolist()])
                distances = torch.cdist(new_vec, new_vectors, p=2)[0]
                if torch.min(distances) < 0.3:
                    min_index = torch.argmin(distances)
                    closest_entity = items[min_index][0]

                    pred_pos = closest_entity[1]
                else:
                    pred_pos = '/O'
                real_pos = w_list[1]
                if real_pos == pred_pos:
                    true += 1
                total += 1
                if real_pos != '/O':
                    total_positive += 1




        line = f.readline()
print("percision: " + str(true/total) + " " + str(true) + " recall: " + str(true/total_positive))
