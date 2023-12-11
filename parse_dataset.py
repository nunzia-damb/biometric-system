import itertools
from math import floor

import numpy as np

PATH = 'data/'
import os

keystrokes = ['102_keystrokes.txt' , '103_keystrokes.txt', '105_keystrokes.txt']
d = {}


# Parsing dei dati
class UserData(object):
    def __init__(self, lines) -> None:
        # list of phrases
        self.phrases = []
        self.train_averages = []
        self.test_averages = []

        self.id = int(lines[0].split('\t')[0])
        self._split_phrases(lines)

    # calculate error rate = how many wrong letters (either in substitution or in addition/subtraction) have been typed
    def calc_error_rate(self, user_in, user_out):
        err_rate = 0
        for i in range(min(len(user_in), len(user_out))):
            if user_in[i] != user_out[i]:
                err_rate += 1
        err_rate += abs((len(user_in) - len(user_out)))
        err_rate /= len(user_in)
        return err_rate

    # method that split lines to get datas about sentences
    def _split_phrases(self, lines):
        def select_data(data):
            press_time = data[5]
            release_time = data[6]
            return int(press_time), int(release_time)

        sentence = -1
        data_line = []
        for l in lines:
            l_split = l.split('\t')
            if int(l_split[1]) != sentence and sentence != -1:
                err_rate = self.calc_error_rate(l_split[2], l_split[3])
                # efficiently put the error rate as first value
                # data_line.append(data_line[0])
                # data_line[0] = err_rate
                data_line.append(err_rate)
                self.phrases.append(data_line)
                data_line = []

            sentence = int(l_split[1])
            l_split = select_data(l_split)
            data_line.append(l_split)

    # calculate average for press_time, release_time
    def calc_avg(self):
        avg = []
        for i in range(0, len(self.phrases) - 1):
            phrase = self.phrases[i]
            avg_press_time = avg_release_time = 0
            error_rate = self.phrases[i][-1]
            for indx in range(0, len(phrase) - 1):
                press, release = phrase[indx]
                avg_press_time += press
                avg_release_time += release
            avg.append([avg_press_time / len(phrase), avg_release_time / len(phrase), error_rate])
        return avg

    def calc_avg_numpy(self):
        import numpy as np
        means = []
        for phrase in self.phrases:
            a = np.array(phrase[:-1])
            means.append(list(np.mean(a, axis=0)))
        return means


# all lines are parsed and the dict is built
for k in keystrokes:
    with open(PATH + k, 'r', encoding='utf-8') as f:
        from time import time

        start_time = time()
        u = UserData(f.readlines()[1:])
        print(u.phrases)
        avgs = u.calc_avg()  # avg_press, avg_release, avg_err
        d[u.id] = avgs
        print('time spent', time() - start_time)
print(d)

# all lines are parsed and the dict is built
for k in keystrokes:
    with open(PATH + k, 'r', encoding='utf-8') as f:
        from time import time

        start_time = time()
        u = UserData(f.readlines()[1:])
        print(u.phrases)
        avgs = u.calc_avg()  # avg_press, avg_release, avg_err
        d[u.id] = avgs
        print('time spent', time() - start_time)
print(d)

# data to use in test and train
test_data = {k: v[floor(len(v) / 2):] for k, v in zip(d.keys(), d.values())}

positive_couples = []
for key, value in d.items():
    # combines 2 features vectors for every user
    positive_couples.extend(list(itertools.combinations(d[key], 2)))

negative_couples = []
keys = list(d.keys())

# combinations of couples from different subjects
for k1 in range(len(keys)):
    for key2 in list(keys)[k1:]:
        # uses product
        key1 = keys[k1]
        if key2 != key1:
            negative_couples += list(
                itertools.product(d[key1], d[key2]))

print(positive_couples, len(positive_couples))
print(negative_couples, len(negative_couples))


train_X1, test_X1 = [], []
train_X2, test_X2 = [], []
train_y, test_y = [], []
for i in range(len(positive_couples)):
    if i < len(positive_couples)//2:
        train_X1.append(positive_couples[i][0])
        train_X2.append(positive_couples[i][1])
        train_y.append(1)
    else:
        test_X1.append(positive_couples[i][0])
        test_X2.append(positive_couples[i][1])
        test_y.append(1)

for i in range(len(negative_couples)):
    if i < len(negative_couples)//2:
        train_X1.append(negative_couples[i][0])
        train_X2.append(negative_couples[i][1])
        train_y.append(0)

    else:
        test_X1.append(negative_couples[i][0])
        test_X2.append(negative_couples[i][1])
        test_y.append(0)

train_y = np.array(train_y).astype(np.float32)
train_X1 = np.array(train_X1).astype(np.float32)
train_X2 = np.array(train_X2).astype(np.float32)

test_y = np.array(test_y).astype(np.float32)
test_X1 = np.array(test_X1).astype(np.float32)
test_X2 = np.array(test_X2).astype(np.float32)
