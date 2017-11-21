'''
Utilities for training a neural network for automated identification of
metaphorical violence.

Author: Matthew A. Turner
Date: 2017-11-21
'''
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
