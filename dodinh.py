'''
Implementation of Do Dinh, E.-L., & Gurevych, I. (2016) using TensorFlow.


Do Dinh, E.-L., & Gurevych, I. (2016). Token-Level Metaphor
    Detection using Neural Networks. Proceedings of the Fourth Workshop on
    Metaphor in NLP, (June), 28–33.

Author: Matthew A. Turner
Date: 2017-11-21
'''
import gensim


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
