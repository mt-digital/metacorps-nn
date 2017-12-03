'''

'''
from uuid import uuid4

# Command-line interface: read it CLIck.
import click
import numpy as np
import pandas as pd
import tensorflow as tf

# See https://radimrehurek.com/gensim/models/keyedvectors.html
import gensim

from eval import Eval
from model import train_network
from util import MetaphorData


# WORKFLOW
# 1. Run ./modelrun with options for how many layers, what else?
# 1. Confirm that X number of simulations will be run.
#
# CLICK EXAMPLE TO MODIFY:
@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    for x in range(count):
        click.echo('Hello %s!' % name)


class ModelRun:
    '''
    Neural network modeling harness to be used in grid searching and general
    evaluation.

    Returns:
        (eval.Eval): evaluation object for comparison with other models
    '''
    def __init__(self,
                 labelled_data_loc='augmented.csv',
                 w2v_model_loc='/data/GoogleNews-vectors-negative300.bin',
                 n_hidden=[300, 150],
                 train_ratio=0.8,
                 validation_ratio=0.1,
                 learning_rate=0.01,
                 batch_size=40,
                 activation='relu',
                 run_directory=str(uuid4())):
        '''

        '''
        self.labelled_data_loc = labelled_data_loc
        self.n_hidden = n_hidden
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.run_directory = run_directory

        try:
            print('loading GoogleNews word2vec embeddings, takes a minute...')
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                w2v_model_loc, binary=True, limit=100000
            )
        except Exception as e:
            print(
                '\n****\nDownload Google News word2vec embeddings from '
                'https://goo.gl/WdCunP to your /data/ directory, dork!\n****\n'
            )
            print(e.message())
            return None

        self.metaphors = MetaphorData(
            labelled_data_loc, self.w2v_model, train_ratio=train_ratio,
            validation_ratio=validation_ratio
        )

        self.train, self.test = self.metaphors.split_train_test()
        # import ipdb
        # ipdb.set_trace()
        X_test = self.test.embedded_sentences
        y_test = self.test.is_metaphor_vec

        X_test, unique_idx = np.unique(X_test, axis=0, return_index=True)
        y_test = y_test[unique_idx]

        self.test.embedded_sentences = X_test
        self.test.is_metaphor_vec = y_test

    def run(self, n_epochs=20):
        # Build checkpoint name based on parameters.
        checkpoint_name = '{}/{}'.format(
            self.run_directory, '-'.join(str(n) for n in self.n_hidden))
        checkpoint_name += '-{}-'.format(self.train_ratio)
        checkpoint_name += '-{}-'.format(self.validation_ratio)
        checkpoint_name += '-{}-'.format(self.learning_rate)
        checkpoint_name += '-{}-'.format(self.activation)

        X, logits = train_network(self.w2v_model, self.train,
                                  checkpoint_name,
                                  n_epochs=n_epochs,
                                  n_hidden=self.n_hidden,
                                  batch_size=self.batch_size,
                                  learning_rate=self.learning_rate)
        saver = tf.train.Saver()

        # X_test = np.array(list(self.test.embedded_sentences))
        # y_test = np.array(list(self.test.is_metaphor_vec))
        # X_test = self.test.embedded_sentences

        import ipdb
        ipdb.set_trace()
        # X_test, unique_idx = np.unique(X_test, axis=0, return_index=True)
        # y_test = y_test[unique_idx]

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_name)
            Z = logits.eval(feed_dict={X: self.test.embedded_sentences})
            y_pred = np.argmax(Z, axis=1)

        import ipdb
        ipdb.set_trace()
        self.test.add_predictions(y_pred)

        return Eval(self.test)


if __name__ == '__main__':
    hello()
