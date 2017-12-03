'''
Following Do Dinh, E.-L., & Gurevych, I. (2016) using TensorFlow. Simplified
in a few respects, but seems to be performing respectably.

Do Dinh, E.-L., & Gurevych, I. (2016). Token-Level Metaphor
    Detection using Neural Networks. Proceedings of the Fourth Workshop on
    Metaphor in NLP, (June), 28–33.

Author: Matthew A. Turner
Date: 2017-11-25
'''
import tensorflow as tf


def train_network(w2v_model, training_data, model_save_path, n_outputs=2,
                  n_hidden=[300], context_window=5, learning_rate=0.5,
                  activation=tf.nn.relu,
                  n_epochs=40, batch_size=50):
    '''
    Arguments:
        w2v_model (gensim.models.word2vec): Gensim wrapper of the word2vec
            model we're using for this training
        training_data (pandas.DataFrame): tabular dataset of training data
        model_save_path (str): location to save model .ckpt file
        n_hidden (list(int)): number of nodes per hidden layer; number of
            elements in list is the number of layers
        context_window (int): number of words before and after focal token
            to consider
        learning_rate (float): stochastic gradient descent learning rate
        n_features (int): number of features for each word embedding
            representation

    Returns:
        Trained network (TODO: what is the type from TF?)
    '''
    # Reset for interactive work?
    tf.reset_default_graph()

    # Total entries in potential metaphor phrase embedding.
    n_sent_tokens = (2 * context_window) + 1
    n_inputs = n_sent_tokens * w2v_model.vector_size

    # Shape of None allows us to pass all of the batch in at once with a
    # variable batch_size.
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    layers = []
    with tf.name_scope('dnn'):

        for idx, n in enumerate(n_hidden):

            name = 'hidden' + str(idx)
            if idx == 0:
                layers.append(
                    tf.layers.dense(
                        X, n, name=name, activation=activation
                    )
                )
            else:
                layers.append(
                    tf.layers.dense(
                        layers[idx - 1], n, name=name, activation=activation
                    )
                )

        logits = tf.layers.dense(layers[-1], n_outputs, name='outputs')

    # Currently this is coming from the TF book Ch 10.
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    X_validate, y_validate = training_data.validation()

    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epochs):

            training_data.shuffle()

            for iteration in range(training_data.num_examples // batch_size):
                X_batch, y_batch = training_data.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_validate, y: y_validate})

            print(epoch, 'Train accuracy: ', acc_train,
                  ' Validation accuracy: ', acc_val)

        saver.save(sess, model_save_path)

    return X, logits
