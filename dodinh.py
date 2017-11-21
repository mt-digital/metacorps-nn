'''
Implementation of Do Dinh, E.-L., & Gurevych, I. (2016) using TensorFlow.


Do Dinh, E.-L., & Gurevych, I. (2016). Token-Level Metaphor
    Detection using Neural Networks. Proceedings of the Fourth Workshop on
    Metaphor in NLP, (June), 28–33.

Author: Matthew A. Turner
Date: 2017-11-21
'''
import gensim
import warnings


def load_w2v_model(embeddings_path):

    return gensim.models.KeyedVectors.load_word2vec_format(
        embeddings_path, binary=True
    )


def train_network(w2v_model, training_data, model_save_path,
                  n_hidden=300, context_window=5, learning_rate=0.5):
    '''
    Arguments:
        w2v_model (gensim.models.word2vec): Gensim wrapper of the word2vec
            model we're using for this training
        training_data (pandas.DataFrame): tabular dataset of training data
        model_save_path (str): location to save model .ckpt file
        n_hidden (int): number of nodes in the hidden layer
        context_window (int): number of words before and after focal token
            to consider

    Returns:
        Trained network (TODO: what is the type from TF?)
    '''

    pass


def test_network(test_data, model_load_path):
    pass


def get_window(text, focal_token, window_size):

    tokens = text.split()
    n_tokens = len(tokens)
    # By default we don't need to enlarge the right window, but we might.
    enlarge_right = False

    # There must be at least 2*window_size + 1 tokens in order to make the
    # window list.
    if n_tokens < (2 * window_size) + 1:
        warnings.warn(
            'window size exceeds number of tokens for text {}'.format(tokens)
        )
        return None

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

    return tokens[left_idx:right_idx]
