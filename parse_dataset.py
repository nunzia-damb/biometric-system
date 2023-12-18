import itertools
from math import floor

import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras.layers
import tensorflow as tf

PATH = '/media/tommy/Volume/Universita/Magistrale/Biometric Systems/project/Keystrokes/files/'
# PATH = '/Users/nunziadambrosio/PycharmProjects/biometric-system/data/'

import os

# fix random seed for reproducibility
np.random.seed(42069)

a = '''103_keystrokes.txt
105_keystrokes.txt'''.split('\n')

b = '''102_keystrokes.txt
103_keystrokes.txt
105_keystrokes.txt
106_keystrokes.txt
109_keystrokes.txt
112_keystrokes.txt
113_keystrokes.txt
114_keystrokes.txt
1002_keystrokes.txt
1005_keystrokes.txt
1010_keystrokes.txt
1011_keystrokes.txt
1015_keystrokes.txt
1020_keystrokes.txt
1024_keystrokes.txt
1027_keystrokes.txt
1030_keystrokes.txt
1047_keystrokes.txt
1048_keystrokes.txt
1053_keystrokes.txt
1054_keystrokes.txt'''.split('\n')
keystrokes = b
d = {}


class DataParser(object):
    def __init__(self, files, *, base_path='', types=None, remove_headers=None, header_injectors=None,
                 normalize_size=None):
        """
        :param files: list of files to parse
        :param base_path: base path for files (final path = base_path + files)
        :param types: list of types to convert the respective index value in the files of the dataset
        :param remove_headers: list of headers names to remove from the dataset
        :param header_injectors: list of HeaderInjector objects to inject into the dataset headers and function to
                extract their values
        :param normalize_size: object SizeNormalization apply padding and truncations
        """
        self.types = [int, int, str, str, int, int, int, str, int] if types is None else types
        self.remove_headers = remove_headers if remove_headers is not None else []
        self.header_injectors = header_injectors if header_injectors is not None else []
        self.files = files
        self.base_path = base_path
        self.user_data = []
        # self.padding_value = padding_value
        # self.max_height = max_height
        self.normalize_size = normalize_size

    # def get_dimension(self):
    #     return len(self.headers)
    def get_shape(self):
        return self.normalize_size.padding.height, len(self.user_data[0].headers)

    def parse(self):
        for file in self.files:
            with open(self.base_path + file, 'r', encoding='utf-8') as fl:
                headers = fl.readline().strip().split('\t')
                lines = fl.readlines()

            ud = UserData(lines, headers=headers, headers_types=self.types)

            # does nothing if there are no headers to remove
            for add_header in self.header_injectors:
                ud = add_header.inject(ud)

            ud = self.remove_headers_columns(headers=headers, user_data=ud)
            if self.normalize_size is not None:
                ud = self.normalize_size(ud)
            self.user_data.append(ud)

        return self.user_data

    def remove_headers_columns(self, user_data, headers):
        remove_headers_indexes = [headers.index(h) for h in self.remove_headers]
        user_data.headers = [h for h in user_data.headers if h not in self.remove_headers]
        if len(remove_headers_indexes) == 0:
            return user_data

        for phrase in user_data.phrases:
            for idx in range(len(phrase)):
                phrase[idx] = [phrase[idx][i] for i in range(len(phrase[idx])) if i not in remove_headers_indexes]
                pass
        return user_data


class UserData(object):
    def __init__(self, lines, *, headers=None, headers_types=None) -> None:
        # list of phrases
        self.headers = [] if headers is None else headers
        self.headers_types = [] if headers_types is None else headers_types
        self.id = int(lines[0].split('\t')[0])
        self.phrases = self._split_phrases(lines)

    def __iter__(self):
        return iter(self.phrases)

    def __eq__(self, other):
        if not isinstance(other, UserData):
            return False
        return other.id == self.id

    def _convert_type(self, line):
        for index in range(len(self.headers_types)):
            if type(line[index]) is not self.headers_types[index]:
                line[index] = self.headers_types[index](line[index])  # magic
        return line

    def _split_phrases(self, lines):
        sentence = -1
        data_line, phrases = [], []
        for index in range(len(lines)):
            line = lines[index]
            l_split = line.split('\t')
            if int(l_split[1]) != sentence and sentence != -1:
                phrases.append(data_line)
                data_line = []
            sentence = int(l_split[1])
            l_split[-1] = l_split[-1].rstrip()  # remove \n
            l_split = self._convert_type(l_split)
            data_line.append(l_split)
        phrases.append(data_line)
        return phrases


class NormalizeDatasetSize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, userdata):
        ud = self.normalize_size(userdata)

        return ud

    def normalize_size(self, userdata):
        """abstract method; template method pattern"""
        return userdata

    # def can_apply(self, phrase_userdata):
    #     """abstract method"""
    #     return


class TruncateUserData(NormalizeDatasetSize):
    def __init__(self, max_height, padding_value):
        super().__init__(max_height)
        self.padding_value = padding_value

    def normalize_size(self, userdata):
        return self.apply_truncate_user_data(userdata)

    def apply_truncate_user_data(self, userdata):
        if self.height < 0:
            return userdata
        for phrase_data in userdata:
            if len(phrase_data) <= self.height:
                # if padding isn't needed
                continue

            trunc_diff = len(phrase_data) - self.height
            for _ in range(trunc_diff):
                phrase_data.pop()

        return userdata


class PaddingUserData(NormalizeDatasetSize):
    def __init__(self, min_height, padding_value):
        super().__init__(min_height)
        self.padding_value = padding_value

    def normalize_size(self, userdata):
        ud = self.apply_padding_user_data(userdata)
        return ud

    def apply_padding_user_data(self, userdata):
        if self.height < 0:
            return userdata
        for i in range(len(userdata.phrases)):
            phrase_data = userdata.phrases[i]
            if len(phrase_data) >= self.height:
                # if padding isn't needed
                continue
            pad_diff = self.height - len(phrase_data)
            phrase_data.extend([[]] * pad_diff)
        return userdata


class SizeNormalization(object):
    def __init__(self, *, padding, truncate):
        self.padding = padding
        self.truncate = truncate

    def __call__(self, userdata):
        return self.normalize_size(userdata)

    def normalize_size(self, userdata):
        userdata = self.truncate(userdata)
        userdata = self.padding(userdata)
        for i in range(len(userdata.phrases)):
            padded_inputs = tf.keras.utils.pad_sequences(userdata.phrases[i], padding="post")
            userdata.phrases[i] = padded_inputs
        return userdata


class HeaderInjector(object):
    def __init__(self, header, func):
        self.headers = header
        self.func = func

    def inject(self, user_data):
        user_data.headers.append(self.headers)
        self.func(user_data)
        return user_data


def add_pp_latency(user_data: UserData):
    press_time_index = user_data.headers.index('PRESS_TIME')
    for phrase in user_data.phrases:
        phrase[0].append(0)
        for line_index in range(1, len(phrase)):
            pp_lat = phrase[line_index][press_time_index] - phrase[line_index - 1][press_time_index]
            phrase[line_index].append(pp_lat)


def add_rp_latency(user_data: UserData):
    press_time_index = user_data.headers.index('PRESS_TIME')
    release_time_index = user_data.headers.index('RELEASE_TIME')
    for phrase in user_data.phrases:
        phrase[0].append(0)
        for line_index in range(1, len(phrase)):
            rp_lat = phrase[line_index][press_time_index] - phrase[line_index - 1][release_time_index]
            phrase[line_index].append(rp_lat)


def add_hold_time(user_data: UserData):
    press_time_index = user_data.headers.index('PRESS_TIME')
    release_time_index = user_data.headers.index('RELEASE_TIME')
    for phrase in user_data.phrases:
        for line in phrase:
            line.append(line[release_time_index] - line[press_time_index])


class CoupleGenerator(object):
    def __init__(self, users_data):
        self.users_data = users_data
        self.positive_couples = []
        self.negative_couples = []

    def split_dataset(self):
        pass

    def generate_positive_couples(self):
        import itertools as it
        couples = []
        for users_data in self.users_data:
            pos_comb = list(it.combinations(users_data, 2))
            couples.extend(pos_comb)
        # couples = [list(it.combinations(users_data, 2)) for users_data in self.users_data]
        self.positive_couples = couples
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
        self.negative_couples = couples
        return couples


from time import time as t

start_time = t()
height_normalization = 70
data_parser = DataParser(keystrokes, base_path=PATH,
                         remove_headers=['TEST_SECTION_ID', 'SENTENCE',
                                         'USER_INPUT', 'KEYSTROKE_ID',
                                         'LETTER', 'PARTICIPANT_ID', 'PRESS_TIME', 'RELEASE_TIME'],
                         header_injectors=[
                             HeaderInjector('HOLD_LATENCY', add_hold_time),
                             HeaderInjector('PP_LATENCY', add_pp_latency),
                             HeaderInjector('RP_LATENCY', add_rp_latency),
                         ],
                         normalize_size=SizeNormalization(
                             truncate=TruncateUserData(max_height=height_normalization, padding_value=-1),
                             padding=PaddingUserData(min_height=height_normalization, padding_value=-1),
                         ),
                         )
data_parser.parse()
# it would be best to normalize before doing couples

cg = CoupleGenerator(data_parser.user_data)
p = cg.generate_positive_couples()
n = cg.generate_negative_couples()

print('time spent', t() - start_time)
y_neg = np.zeros(len(n))
y_pos = np.ones(len(p))
X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(n, y_neg, test_size=1 / 3, random_state=1127)
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(p, y_pos, test_size=1 / 3, random_state=1127)
del y_neg, y_pos, p, n, cg
pass

X_train = X_train_neg + X_train_pos
y_train = np.concatenate((y_train_neg, y_train_pos), axis=0)

X_test = X_test_neg + X_test_pos
y_test = np.concatenate((y_test_neg, y_test_pos), axis=0)
shape = data_parser.get_shape()
# print(len(p)+len(n), len(y))

# data normalization


# print(padded_inputs)


# masked_output_pos = embedding(pos)
# masked_output_neg = embedding(neg)

pass
# norm_pos = mean_zero(p)
# norm_neg = mean_zero(n)
# print(norm_pos, len(norm_pos))
# pos = np.array(norm_pos)
# # print(pos.shape)
# neg = np.array(norm_neg)

# masked_output_pos = embedding(pos)
# masked_output_neg = embedding(neg)
# print(neg[-1], masked_output_neg._keras_mask)
pass
# X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(None, None, test_size=1 / 3, random_state=1127)
# X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(None, None, test_size=1 / 3, random_state=1127)

'''
# every phrase in *positive* couples has length 45
for phrase in p:
    for key in phrase:
        if len(key) > 45:
            key[:] = key[:45]
        elif len(key) < 45:
            for i in range(len(key), 45):
                key.append([0, 0, 0, 0])
'''

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
