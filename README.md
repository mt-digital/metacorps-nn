# MATH 292 Fall 2017 UC Merced -- Final Project

## Led by Prof. Harish Bhat

### Authors: Eshita Nandini and Matt Turner

We aim to see how well neural networks can learn to identify metaphor in corpora. Starting from some gold 
standard training data, the network will perform supervised learning. It will then use new identifications
of metaphor as training data if the system is confident enough in the prediction. If not, the system will
transfer the potential instance of metaphor to a web-based user interface for human confirmation and
editing, if necessary.

This will require a deep understanding of vector representations of words and frames in corpora, as well as 
recurrent neural networks, which are often used for word prediction. We hope to uncover evidence that the
hypothesized conceptual networks of metaphor are expressed by the learned network itself.

We are at the very first stages of development. So far this repository just
contains an edited version of 
[the TensorFlow tutorial's `word2vec_basic.py`](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/)
and [an iPython Notebook](word2vecDemo.ipynb) that shows a high-level usage
of word2vec.
