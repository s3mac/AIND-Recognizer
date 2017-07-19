import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []


    for i in range(test_set.num_items):
        word_propabilities = [(float('-inf'),"")]
        wp={}
        X, lengths = test_set.get_item_Xlengths(i)

        for w,m in models.items():
            try:
                LogL = m.score(X, lengths)

            except:
                LogL = float('-inf')

            word_propabilities.append((LogL,w))
            wp[w] = LogL

        _, best_word = max(word_propabilities)
        guesses.append(best_word)
        probabilities.append(wp)
    return probabilities, guesses
