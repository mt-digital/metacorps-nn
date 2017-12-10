'''
Utilities for training a neural network for automated identification of
metaphorical violence.

Author: Matthew A. Turner
Date: 2017-11-21
'''
import itertools
import numpy as np
import pandas as pd
import random
import warnings


def get_window(text, focal_token, window_size):
    '''
    Given some text and a token of interest, generate a list representing the
    context window that will be classified metaphor or not.

    Arguments:
        text (str): input text from the training or test dataset
        focal_token (str): token to be determined metaphor or not
        window_size (int): half the number of context tokens; ideally this
            will be the number of tokens on either side of the focal_token,
            however the window will adapt if there are not enough words
            to either side of the focal_token

    Returns:
        (list) context tokens and focal_token
    '''
    tokens = text.split()
    n_tokens = len(tokens)
    # By default we don't need to enlarge the right window, but we might.
    enlarge_right = False

    # There must be at least 2*window_size + 1 tokens in order to make the
    # window list. Add
    full_window_len = (2 * window_size) + 1
    if n_tokens < full_window_len:
        tokens.extend('' for _ in range(full_window_len - n_tokens))
        assert len(tokens) == full_window_len
        return tokens

    # We'll get a ValueError if the focal_token was not found in the list.
    try:
        focal_token_idx = next(idx for idx, token in enumerate(tokens)
                               if focal_token in token)  # handle beaten, etc.

    except ValueError as e:
        warnings.warn(e.message)
        return None

    # Handle possibility that the focal token doesn't have window_size
    # number of words ahead of it to use in the window.
    left_padding = focal_token_idx
    if left_padding < window_size:
        left_window_size = left_padding
        enlarge_right = True
    else:
        left_window_size = 5

    # Number of tokens following focal_token in tokens.
    right_padding = n_tokens - focal_token_idx - 1

    if right_padding < window_size:
        right_window_size = right_padding
        # Subtract 2: one for the focal_token and one for zero indexing.
        left_window_size = window_size + (n_tokens - focal_token_idx) - 2
    elif enlarge_right:
        right_window_size = (window_size * 2) - left_padding
    else:
        right_window_size = 5

    left_idx = focal_token_idx - left_window_size
    right_idx = focal_token_idx + right_window_size + 1

    ret = tokens[left_idx:right_idx]

    # XXX Not sure why this is happening, but this should limp us along.
    if len(ret) < full_window_len:
        remaining = full_window_len - len(ret)
        ret.extend('' for _ in range(remaining))
    # XXX Same here, don't know why but this should fix it. Figure this out!
    elif len(ret) > full_window_len:
        ret = ret[:full_window_len]

    return ret


def _make_sentence_embedding(sentence, focal_token, wvmodel, window_size):
    '''
    Use the word2vec model to convert the sentence, or sequence of words,
    to a sequence of embeddings. Limit the number of words/embeddings to
    be window_size * 2 + 1, with focal_token (e.g. attack) at the center.
    This is done by get_window, see that for how windowing is done.

    Arguments:
        sentence (str): sentence that is either a metaphor or not
        focal_token (str): the word that instantiates the source domain,
            e.g. attack
        wvmodel (gensim.models.Word2Vec): vector space model of words as
            gensim model
        window_size (int): ideal number of words before and after focal_token;
            may not be satisfied exactly if token word occurs early or late
            in sentence

    Returns:
        (numpy.ndarray): matrix where each row is an embedding of the word
            in the order they appear in the sentence
    '''
    embed_dim = wvmodel.vector_size
    window = get_window(sentence, focal_token, window_size)

    mat_ret = np.array([
        wvmodel.wv[word]
        if word in wvmodel.wv
        else np.zeros(shape=(embed_dim))
        for word in window
    ])

    # Following MNIST example for now, flattening data. TODO: try using
    # matrices.
    return mat_ret.flatten()


class MetaphorData:

    def __init__(self, data_path, w2v_model, train_ratio=0.8,
                 validation_ratio=0.1, window_size=5):
        '''
        Load labelled metaphor snippets from a .csv file. Provides methods for
        creating batches of training/test data of embeddings from rows of
        the .csv.

        Arguments:
            data_path (str): location of .csv on disk
            w2v_model (gensim.Word2VecModel):
        '''
        self.data_frame = pd.read_csv(data_path)

        # Used to generate sentence embeddings of text to classify.
        self.wv = w2v_model

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = 1 - train_ratio - validation_ratio
        self.window_size = window_size

    def split_train_test(self, **attr_updates):
        '''
        Create a fresh set of training/test data using current attributes as
        parameters.
        '''
        for key in attr_updates:
            setattr(self, key, attr_updates[key])

        _train, _validation, _test = self._split()
        num_train = len(_train)
        # num_validation = len(_validation)
        num_test = len(_test)

        # Build training and validation embeddings.
        train_embeddings = (
            _make_sentence_embedding(row[0], row[1], self.wv, self.window_size)
            for row in _train[['text', 'word']].as_matrix()
        )
        validation_embeddings = (
            _make_sentence_embedding(row[0], row[1], self.wv, self.window_size)
            for row in _validation[['text', 'word']].as_matrix()
        )
        # Wrap training and validation embeddings and their labels with
        # MetaphorDataTrain class.
        self.train = MetaphorDataTrain(
            train_embeddings, _train.is_metaphor.as_matrix(), num_train,
            (validation_embeddings, _validation.is_metaphor)
        )
        # Create the test sentence embeddings.
        test_embeddings = (
            _make_sentence_embedding(row[0], row[1], self.wv, self.window_size)
            for row in _test[['text', 'word']].as_matrix()
        )
        # Initialize the test data object.
        self.test = MetaphorDataTest(
            test_embeddings, _test.is_metaphor.as_matrix(), num_test,
        )
        # Add information about the original sentences, words, and labels.
        self.test.add_original(_test)

        return self.train, self.test

    def _split(self, random_seed=42):

        df = self.data_frame
        n_rows = len(df)
        # n_test is implicitly set when we take set differences below.
        n_train = int(n_rows * self.train_ratio)

        train_indexes = np.random.choice(
            range(n_rows), n_train, replace=False
        )

        n_validation = int(n_train * self.validation_ratio)
        validation_indexes = train_indexes[-n_validation:]
        train_indexes = train_indexes[:-n_validation]

        # Count number of metaphors in training selection.
        train_df = df.iloc[train_indexes]
        # Should be int but this doesn't hurt.
        n_metaphor = int(df.is_metaphor.sum())

        # Need to sample with replacement to build balanced training dataset.
        n_to_sample = n_train - (2 * n_metaphor)
        metaphor_rows = df[df.is_metaphor == 1]

        if random_seed is not None:
            np.random.seed(random_seed)

        sample_indexes = np.random.choice(range(n_metaphor), n_to_sample)
        metaphor_rows.reset_index()

        train_df = train_df.append(
            metaphor_rows.iloc[sample_indexes],
            ignore_index=True
        )

        # This will be random order, length = len(n_rows) - len(n_train).
        test_indexes = list(
            set(self.data_frame.index)
            - set(train_indexes)
            - set(validation_indexes)
        )

        return (
            train_df,
            df.iloc[validation_indexes],
            df.iloc[test_indexes]
        )


class MetaphorDataTrain:

    def __init__(self, embedded_sentences, is_metaphor_vec, num_examples,
                 validation_set=None):
        '''
        Wrap metaphor training data to generate new batches for training and
        validation.

        Arguments:
            embedded_sentences (iter): iterator of 2D matrices representing
                sentences from the corpus
            is_metaphor_vec (pandas.Series or numpy.ndarray): 1D sequence of
                labels corresponding to each embedded sentence; 1 for
                metaphor and 0 is not metaphor
            validation_set (tuple): two-tuple of a list or array of
                sentence embeddings in first position and corresponding
                list or array of 1/0 metaphor/not labels in the second position
        '''
        # Embedded sentences are of dimension window_size * embedding_dim.
        # They will be cycled over each epoch.
        self.num_examples = num_examples
        # Step sizes that will not result in repeated selections of
        # training examples in islice below, i.e. is relative prime of the
        # total number of examples we have. Used to effectively shuffle
        # training examples (see shuffle method below).
        self.suitable_start = [
            i for i in range(1, num_examples) if num_examples % i != 0
        ]
        self.is_metaphor_vec = np.array(list(is_metaphor_vec))
        self.embedded_sentences = np.array(list(embedded_sentences))
        self.embedded_sentences_cycle = \
            itertools.cycle(self.embedded_sentences)
        self.is_metaphor_cycle = itertools.cycle(self.is_metaphor_vec)
        if validation_set is not None:
            self.validation_embeddings = validation_set[0]
            self.validation_labels = validation_set[1]
            self.validation_ = (
                np.array(list(self.validation_embeddings)),
                self.validation_labels
            )

        self.start = 0

    def next_batch(self, batch_size):

        # Randomize in-batch order.
        sel_idx = np.random.permutation(batch_size)
        embed_batch = np.array(list(
            itertools.islice(
                self.embedded_sentences_cycle, self.start,
                self.start + batch_size
            )
        ))[sel_idx]

        is_metaphor_batch = np.array(list(
            itertools.islice(
                self.is_metaphor_cycle, self.start,
                self.start + batch_size
            )
        ))[sel_idx]

        return embed_batch, is_metaphor_batch

    def validation(self):
        return self.validation_

    def shuffle(self):
        self.start = random.choice(self.suitable_start)


class MetaphorDataTest(MetaphorDataTrain):

    def __init__(self, embedded_test_sentences, is_metaphor_vec, num_examples):

        super().__init__(
            embedded_test_sentences, is_metaphor_vec, num_examples
        )

        self.text_sentences = None
        self.predicted_is_metaphor_vec = None

    def add_original(self, test_df):
        '''
        Add the original test dataframe to the object.

        Arguments:
            test_df (pandas.DataFrame): table with original sentences
                and other metadata in same row order as the test
                embeddings and test labels.

        Returns:
            None
        '''
        self.text_sentences = test_df['text']
        self.word = test_df['word']
        self.test_df = test_df

    def add_predictions(self, predicted_is_metaphor_vec, probabilities):
        self.predicted_is_metaphor_vec = predicted_is_metaphor_vec
        self.probabilities = probabilities
