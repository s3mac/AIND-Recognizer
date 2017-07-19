import numpy as np
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit
from asl_utils import test_std_tryit
import warnings
from hmmlearn.hmm import GaussianHMM
import math
from matplotlib import (cm, pyplot as plt, mlab)
import timeit
from my_model_selectors import SelectorConstant
from my_model_selectors import SelectorCV
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
import unittest
from unittest import TestCase

from my_recognizer import recognize
from asl_utils import show_errors
import logging
import sys

# initializes the database
asl = AslDb()

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

# features for polar coordinate values where the nose is the origin
# 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# 'polar-rr' and 'polar-rtheta' refer to the radius and angle

asl.df['polar-rr'] = (asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2)**0.5
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'],asl.df['grnd-ry'])
asl.df['polar-lr'] = (asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2)**0.5
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'],asl.df['grnd-ly'])

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

# features for normalized by speaker values of left, right, x, y
# 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# Z-score scaling (X-Xmean)/Xstd

asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])

asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

# features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(method='backfill')
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(method='backfill')
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(method='backfill')
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(method='backfill')

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


asl.df['delta-rx-norm'] = asl.df['norm-rx'].diff().fillna(method='backfill')
asl.df['delta-ry-norm'] = asl.df['norm-ry'].diff().fillna(method='backfill')
asl.df['delta-lx-norm'] = asl.df['norm-lx'].diff().fillna(method='backfill')
asl.df['delta-ly-norm'] = asl.df['norm-ly'].diff().fillna(method='backfill')

features_delta_norm = ['delta-rx-norm', 'delta-ry-norm', 'delta-lx-norm', 'delta-ly-norm']


asl.df['polar_rr_mean'] = asl.df['speaker'].map(df_means['polar-rr'])
asl.df['polar_rl_mean'] = asl.df['speaker'].map(df_means['polar-lr'])
asl.df['polar_rtheta_mean'] = asl.df['speaker'].map(df_means['polar-rtheta'])
asl.df['polar_ltheta_mean'] = asl.df['speaker'].map(df_means['polar-ltheta'])

asl.df['polar_rr_std'] = asl.df['speaker'].map(df_std['polar-rr'])
asl.df['polar_rl_std'] = asl.df['speaker'].map(df_std['polar-lr'])
asl.df['polar_rtheta_std'] = asl.df['speaker'].map(df_std['polar-rtheta'])
asl.df['polar_ltheta_std'] = asl.df['speaker'].map(df_std['polar-ltheta'])

asl.df['polar-rr-norm'] = (asl.df['polar-rr'] - asl.df['polar_rr_mean']) / asl.df['polar_rr_std']
asl.df['polar-lr-norm'] = (asl.df['polar-lr'] - asl.df['polar_rl_mean']) / asl.df['polar_rl_std']
asl.df['polar-rtheta-norm'] = (asl.df['polar-rtheta'] - asl.df['polar_rtheta_mean']) / asl.df['polar_rtheta_std']
asl.df['polar-ltheta-norm'] = (asl.df['polar-ltheta'] - asl.df['polar_ltheta_mean']) / asl.df['polar_ltheta_std']

features_polar_norm = ['polar-rr-norm', 'polar-rtheta-norm', 'polar-lr-norm', 'polar-ltheta-norm']

# Define list named 'features_custom' for building the training set
features_custom = features_delta_norm + features_polar_norm




def train_a_word(word, num_hidden_states, features):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict


def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()


def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

"""
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
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

"""
from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)
"""

"""
demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

show_model_stats(demoword, model)

visualize(demoword, model)
"""

# displays the first five rows of the asl database, indexed by video and frame
#print(asl.df.head())

# look at the data available for an individual frame
#print(asl.df.ix[98,1])

# test the code
#test_features_tryit(asl)

#show a single set of features for a given (video, frame) tuple
#print([asl.df.ix[98,1][v] for v in features_ground])

#print(asl.df)
sys.stdout = open('test_output_final.txt', 'w')

#SelectorConstant, SelectorCV, SelectorDIC,
model_selector_list = [ SelectorBIC]
#features_norm, features_polar, features_delta, 
features_list = [features_custom]
results = {"Model Selector":[], "Feature":[], "Time":[], "WER": [], "Correct": [], "Total": []}

for model_selector in  model_selector_list:
    for feature in features_list:

        print("Using feature - {} and model - {} ".format(feature, model_selector))
        start = timeit.default_timer()
        models = train_all_words(feature, model_selector)
        test_set = asl.build_test(feature)
        probabilities, guesses = recognize(models, test_set)
        end = timeit.default_timer()-start

        print("\n Time = {}".format(end))
        S = 0
        N = len(test_set.wordlist)
        num_test_words = len(test_set.wordlist)
        if len(guesses) != num_test_words:
            print("Size of guesses must equal number of test words ({})!".format(num_test_words))
        for word_id in range(num_test_words):
            if guesses[word_id] != test_set.wordlist[word_id]:
                S += 1
        wer = float(S) / float(N)
        correct = N - S
        print("\n**** WER = {}".format(wer))
        print("Total correct: {} out of {}".format(correct, N))
        print('Video  Recognized                                                    Correct')
        print('=====================================================================================================')
        for video_num in test_set.sentences_index:
            correct_sentence = [test_set.wordlist[i] for i in test_set.sentences_index[video_num]]
            recognized_sentence = [guesses[i] for i in test_set.sentences_index[video_num]]
            for i in range(len(recognized_sentence)):
                if recognized_sentence[i] != correct_sentence[i]:
                    recognized_sentence[i] = '*' + recognized_sentence[i]
            print('{:5}: {:60}  {}'.format(video_num, ' '.join(recognized_sentence), ' '.join(correct_sentence)))

        results["Model Selector"].append("{}".format(model_selector))
        results["Feature"].append("{}".format(feature))
        results["Time"].append(end)
        results["WER"].append(wer)
        results["Correct"].append(correct)
        results["Total"].append(N)

        df_results = pd.DataFrame(results)
        df_results.to_csv('results_final.csv')
        print(df_results)
