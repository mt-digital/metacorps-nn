'''
Code to evaluate a particular trained network. Think about formatting results
here well to be tables in the paper.
'''
import sklearn.metrics as skmetrics

from collections import Counter


class Eval:
    '''
    Methods to evaluate different models.
    '''
    def __init__(self, test_data):
        '''
        Arguments:
            test_data (util.MetaphorDataTest): test data
        '''
        self.test_data = test_data

        self.test_sentences = self.test_data.text_sentences

        self.false_negatives = self.test_sentences.as_matrix()[
            (self.test_data.predicted_is_metaphor_vec == 0) &
            (self.test_data.is_metaphor_vec == 1)
        ]
        self.false_positives = self.test_sentences.as_matrix()[
            (self.test_data.predicted_is_metaphor_vec == 1) &
            (self.test_data.is_metaphor_vec == 0)
        ]
        self.true_negatives = self.test_sentences.as_matrix()[
            (self.test_data.predicted_is_metaphor_vec == 0) &
            (self.test_data.is_metaphor_vec == 0)
        ]
        self.true_positives = self.test_sentences.as_matrix()[
            (self.test_data.predicted_is_metaphor_vec == 1) &
            (self.test_data.is_metaphor_vec == 1)
        ]

    def word_counts(self, n_most_common=10):
        '''
        For both the false negatives and false positives, create word counts

        Returns:
            (dict): keyed by false_{negative,positive} true_{negative,positive}
                with word counts as values
        '''
        return {
            'false_negatives':
                Counter(self.false_negatives).most_common(n_most_common),
            'false_positives':
                Counter(self.false_positives).most_common(n_most_common),
            'true_negatives':
                Counter(self.true_negatives).most_common(n_most_common),
            'true_positives':
                Counter(self.true_positives).most_common(n_most_common)
        }

    def confusion_matrix(self):
        '''
        Returns:
            (numpy.ndarray): 2D confusion matrix using true/pred is_metaphor

        Example:
            >>> eval_inst = Eval(test_data)
            >>> tp, fp, fn, tn = eval_isnt.confusion_matrix()
        '''
        pred = self.test_data.predicted_is_metaphor_vec
        true = self.test_data.is_metaphor_vec

        return skmetrics.confusion_matrix(true, pred)
