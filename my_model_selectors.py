import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        scores = [(float('inf'),0)]

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                n = n_components
                f = self.X.shape[1]
                p = n*n+2*n*f -1
                logN = np.log(self.X.shape[0])
                BIC_score = -2 * logL + p * logN
                scores.append((BIC_score,n_components))

            except:
                pass

        _, best_num_components=min(scores)


        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        scores_n_components = [(float('-inf'),0)]

        M = len((self.words).keys())


        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogLi = hmm_model.score(self.X, self.lengths)
                SumLogL = 0
                for word in self.hwords.keys():
                    if word != self.this_word:
                        word_X, word_length = self.hwords[word]
                        SumLogL += hmm_model.score(word_X, word_length)

                DIC_score = LogLi - (1 / (M - 1)) * SumLogL
                scores_n_components.append((DIC_score, n_components))
            except:
                pass

        _, best_n_components = max(scores_n_components)
        return self.base_model(best_n_components) #if best_n_components > 0 else None

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        scores = [(float('-inf'),0)]
        n_splits = 3
        split_method = KFold(random_state=self.random_state, n_splits=n_splits)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                cv_scores=[]
                if(len(self.sequences) < n_splits):
                    return self.base_model(self.n_constant)

                for train_index, test_index in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(train_index, self.sequences)
                    X_test,  lengths_test  = combine_sequences(test_index, self.sequences)


                    hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    logL = hmm_model.score(X_test, lengths_test)


                    cv_scores.append(logL)

                cv_scores_mean = (np.array(cv_scores)).mean()
                scores.append((cv_scores_mean, n_components))

            except:
                pass

        _, best_num_components = max(scores)



        return self.base_model(best_num_components)
