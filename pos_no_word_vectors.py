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
true = 0
total = 0
with open("./data/ass1-tagger-dev") as f:
    line = f.readline()
    c = 0
    while line != '':

        sent_list = line.split(' ')
        for w in sent_list:
            w_list = w.split('/')
            if w_list[0] != '.\n':
                if w_list[0] in words.keys():
                    real_pos = w_list[1]
                    pred_pos = max(words[w_list[0]].counts, key=words[w_list[0]].counts.get)
                    if real_pos == pred_pos:
                        true += 1

                else:
                    real_pos = w_list[1]
                    pred_pos = max(pos_count, key=pos_count.get)
                    if real_pos == pred_pos:
                        true += 1
                total += 1

        line = f.readline()
print("accuracy: " + str(true/total) + " " + str(true) + " correct out of " + str(total))

with open("POS_preds_1.txt", "w") as f1:
    with open("./data/ass1-tagger-test-input") as f2:
        line = f2.readline()

        while line != '':
            new_line = []
            sent_list = line.split(' ')
            for w in sent_list:
                if w != '.\n':
                    if w in words.keys():
                        pred_pos = max(words[w].counts, key=words[w].counts.get)
                        new_line.append('/'.join([w, pred_pos]))
                    else:
                        real_pos = w_list[1]
                        pred_pos = max(pos_count, key=pos_count.get)
                        new_line.append('/'.join([w, pred_pos]))

            new_line.append('./.\n')
            f1.write(' '.join(new_line))
            line = f2.readline()
