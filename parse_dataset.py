import itertools
from math import floor

import numpy as np

PATH = '/media/tommy/Volume/Universita/Magistrale/BiometricSystems/project/Keystrokes/KeyboardKeystrokes/Keystrokes/files/'
import os

# fix random seed for reproducibility
np.random.seed(42069)

keystrokes = '''100390_keystrokes.txt
100395_keystrokes.txt
100396_keystrokes.txt
100397_keystrokes.txt
100410_keystrokes.txt
100416_keystrokes.txt
100417_keystrokes.txt
100419_keystrokes.txt
100420_keystrokes.txt
100421_keystrokes.txt
100422_keystrokes.txt
100423_keystrokes.txt
100425_keystrokes.txt
100426_keystrokes.txt
100431_keystrokes.txt
100432_keystrokes.txt
100434_keystrokes.txt
100438_keystrokes.txt
100439_keystrokes.txt
100444_keystrokes.txt
100445_keystrokes.txt
100446_keystrokes.txt
100447_keystrokes.txt'''.split('\n')
d = {}


class DataParser(object):
    def __init__(self, files, *, base_path='', types=None):
        self.types = [int, int, str, str, int, int, int, str, int] if types is None else types
        self.files = files
        self.base_path = base_path
        self.user_data = []

    def parse(self):
        for file in self.files:
            with open(self.base_path + file, 'r', encoding='utf-8') as fl:
                headers = fl.readline().strip().split('\t')
                lines = fl.readlines()
                self.user_data.append(UserData(lines, headers=headers, headers_types=self.types))


# Parsing dei dati
class UserData(object):
    def __init__(self, lines, *, headers=None, headers_types=None, add_difference_press_release=True) -> None:
        # list of phrases
        self.headers = [] if headers is None else headers
        self.headers_types = [] if headers_types is None else headers_types
        self.phrases = []
        self.train_averages = []
        self.test_averages = []

        self.id = int(lines[0].split('\t')[0])
        self._split_phrases(lines)
        if add_difference_press_release:
            self._add_difference_press_release()

    def _add_difference_press_release(self):
        press_time_index = self.headers.index('PRESS_TIME')
        release_time_index = self.headers.index('RELEASE_TIME')
        for phrase in self.phrases:
            for line in phrase:
                line.append(line[release_time_index] - line[press_time_index])

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
    def _convert_type(self, line):
        for index in range(len(self.headers_types)):
            if type(line[index]) is not self.headers_types[index]:
                line[index] = self.headers_types[index](line[index])
        return line

    def _split_phrases(self, lines):
        sentence = -1
        data_line = []
        for line in lines:
            l_split = line.split('\t')
            if int(l_split[1]) != sentence and sentence != -1:
                self.phrases.append(data_line)
                data_line = []

            sentence = int(l_split[1])
            l_split[-1] = l_split[-1].rstrip()  # remove \n
            l_split = self._convert_type(l_split)
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

from time import time

start_time = time()
data_parser = DataParser(keystrokes, base_path=PATH)
data_parser.parse()
print(data_parser.user_data)
print('time spent', time() - start_time)
print(d)


class CoupleGenerator(object):
    def __init__(self, users_data):
        self.users_data = users_data

    def split_dataset(self):
        pass

    def generate_positive_couples(self):
        import itertools as it

        pass

    def generate_negative_couples(self):
        pass


'''# data to use in test and train
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
    if i < len(positive_couples) // 2:
        train_X1.append(positive_couples[i][0])
        train_X2.append(positive_couples[i][1])
        train_y.append(1)
    else:
        test_X1.append(positive_couples[i][0])
        test_X2.append(positive_couples[i][1])
        test_y.append(1)

for i in range(len(negative_couples)):
    if i < len(negative_couples) // 2:
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

pass
'''