import itertools
from math import floor

import numpy as np
from sklearn.model_selection import train_test_split

PATH = '/media/tommy/Volume/Universita/Magistrale/BiometricSystems/project/Keystrokes/KeyboardKeystrokes/Keystrokes/files/'
# PATH = './data/'

import os

# fix random seed for reproducibility
np.random.seed(42069)

keystrokes = '''103_keystrokes.txt'''.split('\n')
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
    def __init__(self, files, *, base_path='', types=None, remove_headers=None):
        self.types = [int, int, str, str, int, int, int, str, int] if types is None else types
        self.remove_headers = remove_headers if remove_headers is not None else []
        self.files = files
        self.base_path = base_path
        self.user_data = []

    def parse(self):
        for file in self.files:
            with open(self.base_path + file, 'r', encoding='utf-8') as fl:
                headers = fl.readline().strip().split('\t')
                lines = fl.readlines()

                ud = UserData(lines, headers=headers, headers_types=self.types)

                # does nothing if there are no headers to remove
                ud = self.remove_headers_columns(headers=headers, user_data=ud)
                self.user_data.append(ud)

    def remove_headers_columns(self, user_data, headers):
        remove_headers_indexes = [headers.index(h) for h in self.remove_headers]
        if len(remove_headers_indexes) == 0:
            return user_data

        for phrase in user_data.phrases:
            for idx in range(len(phrase)):
                phrase[idx] = [phrase[idx][i] for i in range(len(phrase[idx])) if i not in remove_headers_indexes]
                pass
        return user_data


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
            self.headers.append('DIFF_PRESS_RELEASE')
            self._add_difference_press_release()

    def __iter__(self):
        return iter(self.phrases)

    def __eq__(self, other):
        if not isinstance(other, UserData):
            return False
        return other.id == self.id

    def _add_difference_press_release(self):
        press_time_index = self.headers.index('PRESS_TIME')
        release_time_index = self.headers.index('RELEASE_TIME')
        for phrase in self.phrases:
            for line in phrase:
                line.append(line[release_time_index] - line[press_time_index])

    # calculate error rate = how many wrong letters (either in substitution or in addition/subtraction) have been typed

    # method that split lines to get datas about sentences
    def _convert_type(self, line):
        for index in range(len(self.headers_types)):
            if type(line[index]) is not self.headers_types[index]:
                line[index] = self.headers_types[index](line[index])  # magic
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


# all lines are parsed and the dict is built

class CoupleGenerator(object):
    def __init__(self, users_data):
        self.users_data = users_data

    def split_dataset(self):
        pass

    def generate_positive_couples(self):
        import itertools as it
        couples = []
        for users_data in self.users_data:
            pos_comb = list(it.combinations(users_data, 2))
            couples.extend(pos_comb)
        # couples = [list(it.combinations(users_data, 2)) for users_data in self.users_data]
        return couples

    def generate_negative_couples(self):
        import itertools as it
        couples = []
        for user_data1 in self.users_data:
            for user_data2 in self.users_data:
                if user_data1 == user_data2:
                    continue
                neg_comb = list(it.product(user_data1, user_data2))
                couples.extend(neg_comb)
        # couples = [list(it.product(user_data1, user_data2)) for user_data1 in self.users_data for user_data2 in
        #            self.users_data if user_data1 != user_data2]
        return couples


from time import time

start_time = time()
data_parser = DataParser(keystrokes, base_path=PATH, remove_headers=['TEST_SECTION_ID', 'SENTENCE',
                                                                     'USER_INPUT', 'KEYSTROKE_ID', 'LETTER'])
data_parser.parse()

cg = CoupleGenerator(data_parser.user_data)
p = cg.generate_positive_couples()
n = cg.generate_negative_couples()

print('time spent', time() - start_time)

X_train, X_test, y_train, y_test = train_test_split(None, None, test_size=1 / 3, random_state=1127)

print(d)

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
