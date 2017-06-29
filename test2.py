from my_model_selectors import SelectorBIC
import numpy as np
import pandas as pd
from asl_data import AslDb
import hmmlearn
from sklearn.model_selection import KFold

import timeit

asl = AslDb()

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']


training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
sequences = training.get_all_sequences()
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
"""
print('sequences:')
print(sequences['CHOCOLATE'])
Xlengths = training.get_all_Xlengths()
print('Xlengths')
print(Xlengths['CHOCOLATE'])
print(len(sequences))
split_method = KFold()

a=split_method.split(sequences['CHOCOLATE'])
print(a)

for cv_train_idx, cv_test_idx in a:
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
"""

"""
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))

"""
