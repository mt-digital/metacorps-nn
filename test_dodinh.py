from dodinh import get_window

from nose.tools import eq_


def test_get_window():
    word = 'attack'
    window_size = 5

    # Test when we have enough space on both sides for full window.
    text = 'he has to go on the attack if he wants to win the debate'
    window = get_window(text, word, window_size)
    eq_(window, ['has', 'to', 'go', 'on', 'the', 'attack',
                 'if', 'he', 'wants', 'to', 'win'])

    # Test when there is not enough space on the left.
    text = 'i think romney beat obama in the first debate but obama won the last two'
    word = 'beat'
    window = get_window(text, word, window_size)
    eq_(window, ['i', 'think', 'romney', 'beat', 'obama', 'in', 'the',
                 'first', 'debate', 'but', 'obama'])

    # Test when there is not enough space on the right.
    text = 'although the i think youre right its important to consider the beating taken by romney'
    window = get_window(text, word, window_size)
    eq_(window, ['youre', 'right', 'its', 'important', 'to', 'consider',
                 'the', 'beating', 'taken', 'by', 'romney'])

    text = 'he beat his opponent'
    window = get_window(text, word, window_size)
    eq_(window, None)
