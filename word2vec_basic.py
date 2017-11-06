# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Basic word2vec example.

Edited from the original TensorFlow example
(https://github.com/tensorflow/models/tutorials/embedding/word2vec_basic.py)
to be more clear and compartmentalized for Jupyter NB exposition
by Matt Turner <maturner01@gmail.com>. Also I don't care about Python 2, so
I'm removing compatibility stuff.

It's also specialized to my metaphorical violence detection project on
cable news, so there are defaults for interesting words/concepts.

Date: 2 Nov 2017
"""
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile

from urllib.request import urlopen

DEFAULT_VIOMET_WORDS = ['trump', 'clinton', 'conservative', 'liberal']


def demo_word2vec(network, vocabulary_size=50000,
                  batch_size=128, num_skips=2, skip_window=4,
                  embedding_size=128, num_sampled=128,
                  num_steps=10001, learning_rate=0.2):
    '''
    Simple wrapper to set up and run word2vec
    '''
    print('reading data')
    vocabulary = load_data(network)
    full_vocabulary_size = len(np.unique(vocabulary))

    vocabulary_size = min((vocabulary_size, full_vocabulary_size))
    print('building dataset')
    data, count, dictionary, reverse_dictionary = build_dataset(
        vocabulary, vocabulary_size
    )
    del vocabulary

    print('generating batches')
    batch, labels = generate_batch(data, count, dictionary,
                                   reverse_dictionary)

    print('running word2vec')
    final_embeddings = run_word2vec(
        data, count, dictionary,
        reverse_dictionary, vocabulary_size,
        batch_size=batch_size,
        num_skips=num_skips,
        skip_window=skip_window,
        num_sampled=num_sampled,
        embedding_size=embedding_size,
        num_steps=num_steps,
        learning_rate=learning_rate
    )

    return final_embeddings, dictionary, reverse_dictionary


def load_data(network):

    assert network in ['MSNBCW', 'CNNW', 'FOXNEWSW']

    filename = network + '-2016.txt.zip'

    if not os.path.exists(filename):
        url = 'http://metacorps.io/static/data/' + filename
        open(filename, 'w+').write(
            urlopen(url).read()
        )
        print('{} corpus file {} not found. Downloading from {}'.format(
            network, filename, url
        ))
    else:
        print('{} corpus file {} found, no need to download.'.format(
            network, filename
        ))

    with zipfile.ZipFile(filename) as f:
        ret = tf.compat.as_str(f.read(f.namelist()[0])).split()

    return ret


#: used as a static index in generate_batch
DATA_INDEX = 0


def generate_batch(data, count, dictionary, reverse_dictionary,
                   batch_size=8, num_skips=2, skip_window=1):
    '''
    Still confused about just what a batch is, but hopefully coding it out
    will help me understand
    '''
    # this is a trick to have a "static" counter for generate_batch
    global DATA_INDEX

    # A batch is a set of target words and windows. The number of skips must
    # evenly divide the batch size, otherwise a target word will not have
    # an even number of skips
    assert batch_size % num_skips == 0

    assert num_skips <= 2 * skip_window, \
        'skip window is taken on each side of target word, so num_skips ' \
        'must be twice the skip window'

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer_ = collections.deque(maxlen=span)
    if DATA_INDEX + span > len(data):
        DATA_INDEX = 0

    buffer_.extend(data[DATA_INDEX:DATA_INDEX + span])

    DATA_INDEX += span

    for i in range(batch_size // num_skips):

        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)

        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer_[skip_window]
            labels[i * num_skips + j, 0] = buffer_[context_word]
        if DATA_INDEX == len(data):
            buffer_[:] = data[:span]
            DATA_INDEX = span
        else:
            buffer_.append(data[DATA_INDEX])
            DATA_INDEX += 1

    # Backtrack to avoid skipping words in the end of a batch
    DATA_INDEX = (DATA_INDEX + len(data) - span) % len(data)

    return batch, labels


def build_dataset(vocabulary, vocabulary_size):
    # Ordered dictionary with words as keys and counts as values; see
    # https://docs.python.org/3/library/collections.html#collections.Counter
    count = collections.Counter(vocabulary).most_common()

    dictionary = {
        word_count_pair[0]: index
        for index, word_count_pair in enumerate(count)
    }

    data = [dictionary.get(word, 0) for word in vocabulary]

    reversed_dictionary = {value: key for key, value in dictionary.items()}

    return data, count, dictionary, reversed_dictionary


def run_word2vec(
            data, count, dictionary, reverse_dictionary,
            vocabulary_size, batch_size=128,
            embedding_size=128,  # Dimension of the embedding vector.
            skip_window=1,  # How many words to consider left and right.
            num_skips=2,  # Num times to reuse an input to gen a label.
            learning_rate=1.0,  # SGG step size.
            num_steps=10001,  # Number of training steps
            num_sampled=64,  # Number of negative examples to sample.
            words_to_test=DEFAULT_VIOMET_WORDS,
            verbose=True
        ):
    '''
    Yes.
    '''
    # Get the indices of the example words we'll display below.
    example_indices = np.array([dictionary[word] for word in words_to_test])
    num_examples = len(example_indices)

    # Protect against too small a vocabulary...It's cable news after all.
    vocabulary_size = min(len(dictionary.keys()), vocabulary_size)

    # Initialize a fresh computation graph.
    graph = tf.Graph()

    # Graphs can be used as Python contexts.
    with graph.as_default():

        # Input data placeholders in computational graph.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Need to wrap example index list for inclusion in comp graph.
        example_indices_tf = tf.constant(example_indices, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # Declare look-up embeddings for inputs.
            embeddings = tf.Variable(
                # Initialize the embedding matrix.
                # Rows are words, columns are learned "features".
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Declare NCE loss. Weight matrix has same shape as embedding.
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0/math.sqrt(embedding_size))
            )
            # Not sure yet what the biases or weights do, but note the dim.
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Good explanation of NCE in [1], gives other uses than NLP.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size)
        )

        optimizer = \
            tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Cosine similarity is our heuristic to estimate word relatedness
        # https://en.wikipedia.org/wiki/Cosine_similarity#Properties
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        # embeddings and norm are both tf.Variables
        normalized_embeddings = embeddings / norm

        example_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, example_indices_tf
        )
        # Final step in cosine similarity calculation
        similarity = tf.matmul(
            example_embeddings, normalized_embeddings, transpose_b=True
        )

        # Create variable initializer for variables created in this context.
        # XXX not sure it's necessary to do so given how I'm rewriting this
        init = tf.global_variables_initializer()

        # Begin training.
        with tf.Session(graph=graph) as session:

            init.run()

            for step in range(num_steps):
                batch_inputs, batch_labels = generate_batch(
                    data, count, dictionary, reverse_dictionary,
                    batch_size, num_skips, skip_window
                )
                # Apparently train_{inputs, labels} are hashable.
                feed_dict = {train_inputs: batch_inputs,
                             train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss],
                                          feed_dict=feed_dict)
                if verbose:
                    _print_stats(loss_val, step, similarity,
                                 num_examples, reverse_dictionary,
                                 example_indices)

            final_embeddings = normalized_embeddings.eval()

    return final_embeddings


def _print_stats(loss_val, step, similarity, num_examples,
                 reverse_dictionary, example_indices, top_k=8):

    average_loss = 0
    average_loss += loss_val

    if step % 2000 == 0:
        if step > 0:
            # Estimate loss over last 2000 steps
            average_loss /= 2000
        print(
            'Average loss at step ', step, ': ', average_loss
        )
        average_loss = 0
    if step % 4000 == 0:
        sim = similarity.eval()
        for i in range(num_examples):
            example_word = reverse_dictionary[example_indices[i]]
            # Number of most similar to find
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to {}:'.format(example_word)

            # Print all top_k closest words.
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                if k < top_k - 1:
                    log_str = \
                        '{} {},'.format(log_str, close_word)
                else:
                    log_str = \
                        '{} {}'.format(log_str, close_word)
            print(log_str)


''' References

 [1]  http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
'''
