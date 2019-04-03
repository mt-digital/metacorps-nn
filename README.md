# MATH 292 Fall 2017 UC Merced -- Final Project

## Led by Prof. Harish Bhat

### Author: Matt Turner

We aim to see how well neural networks can learn to identify 
metaphor in corpora. Specifically, this system will first be used to identify
metaphorical violence on television news to support an ongoing project.
Starting from some gold 
standard training data, the network will perform 
supervised learning. It will then use new identifications
of metaphor as training data if the system is confident enough in the 
prediction. If not, the system will transfer the potential instance of 
metaphor to a web-based user interface for human confirmation and editing, 
if necessary.

## Usage

### On the `MERCED` cluster

The numerical experiments were performed on the MERCED cluster at UC Merced.
The runner script workflow is at the prototype stage, but working. Here's how
to use it.

To submit twenty jobs with a learning rate of 0.1

```
./run-trials.sh 0.1
```

Here's `run-trials.sh`:



### Create and evaluate a model

The `ModelRun` class in `modelrun.py` enables hyperparameter tuning and model
evaluation via the `Eval` class in `eval.py`. When a new model is run, a
checkpoint is saved with its parameter settings as the filename once the 
model has completed its training.
Each instance of `ModelRun` gets its own UUID-by-default directory in the 
`modelruns` directory. So this is our rudimentary hyperparameter tuning tool.

Example (see [modelrun.py](/modelrun.py) for more):

```python

from modelrun import ModelRun

# Set up a model run with one 300-node hidden layer, otherwise default args.
mr = ModelRun(n_hidden=[300], limit_word2vec=200000)  # limit word vec dct size
eval1 = mr.run(n_epochs=40)

# Now set model to have two hidden layers, with 300 and 150 nodes each.
mr.n_hidden = [300, 150]
eval2 = mr.run()

# Change activation function. Default is ReLU. Still [300, 150] hidden layers.
mr.activation = tf.nn.selu
eval_selu = mr.run()

# Calculate a number of metrics for these model tests.
print(eval_selu.accuracy)
print(eval_selu.precision_score)
print(eval_selu.recall_score)
print(eval_selu.confusion_matrix)
```

### Loading data

```python
import gensim
import tensorflow as tf

from model import MetaphorData, train_network

data_path = 'augmented_2012.csv'
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    '/Users/mt/Desktop/GoogleNews-vectors-negative300.bin', binary=True
)
train_ratio = 0.8  # fraction of csv to use as training data
validation_ratio = 0.1  # fraction of training data to use for validation
metaphors = MetaphorData(
    data_path, w2v_model, train_ratio=train_ratio,
    validation_ratio=validation_ratio
)

train, test = metaphors.split_train_test()

# Probably a better way to load these variables/tensors from disk, but I'm 
# confused as to how. This seems to work OK. This three-layer configuration
# worked pretty well, and is an interesting example. More hyperparam tuning
# needs to be done.
checkpoint = 'simple_network.ckpt'
X, logits = train_network(w2v_model, train, checkpoint,
                          n_hidden=[200, 100, 50], batch_size=40,
                          learning_rate=0.01)

saver = tf.train.Saver()
# This needs improvement, but works for now.
X_test = np.array(list(test.embedded_sentences))
y_test = np.array(list(test.is_metaphor_vec))
# Need to take unique X_test because there are repeats. This was done because
# I needed to sample with replacement to build training/validation set (mistake
# for validation I guess!), then used 20% of these to build test set.
# Yes this actually seems to work, but I'm not totally convinced yet, needs
# to be fixed to really know what's going on.
X_test, unique_idx = np.unique(X_test, axis=0, return_index=True)
y_test = y_test[unique_idx]

# Test the model!
with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    Z = logits.eval(feed_dict={X: X_test})
    y_pred = np.argmax(Z, axis=1)

# Use sklearn to evaluate performance, with one example set of results shown.
import sklearn.metrics as skmetrics
print(skmetrics.classification_report(y_test, y_pred))
# precision    recall  f1-score   support
#
#          0       0.90      0.87      0.88       440
#          1       0.86      0.89      0.87       388
#
# avg / total       0.88      0.88      0.88       828

print(skmetrics.confusion_matrix(y_test, y_pred))
# [[ tp=384  fp=56]
#  [ fn=44   tn=344]]
```

Like in the [mnist training data](https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py) 
as provided by TensorFlow, the
newly created `train` instance has a `next_batch` that takes `batch_size` as
an argument. The [`MetaphorDataTrain`](/util.py#L163) class wraps 
iterators that cycle through all available training data using a combination
of Python standard library `itertools.cycle` and `itertools.islice`, the latter
selects the next `batch_size` training examples. 

## Data pipeline

The TV News data comes from the 
[Internet Archive's TV News Archive (TVNA)](http://archive.org/tv/details).
The TVNA contains millions of episodes of cable news with video and closed
captions. We have made transcripts out of the closed captioning for all the
shows of interest, and saved these with metadata in a MongoDB instance. 

The script [prepare_csv_input.py](/prepare_csv_input.py) creates a CSV of
a balanced number of training rows. I am wrongly using subsets of these
for validation and tests. For validation it's not that bad, but for test
we need to take only unique input data, as shown in the example above.

The entries of the `text` column of this data are windowed down according
to a `window_size` parameter, where the window size ideally extends before
and after the target word that instantiates the metaphor; at this point we
have target words _attack_, _hit_, and _beat_. 
Then we create a matrix representation of the sequence of `(2*window_size) + 1`
tokens using word embeddings from the Google News word embeddings 
([1.65 GB download](https://doc-08-3o-docs.googleusercontent.com/docs/securesc/nsvfebu7ik236iadibqld9mq9669rtgt/vgklkerbosnf2rf2ic56jvo3rqime6rd/1511042400000/06848720943842814915/13496840407080918705/0B7XkCwpI5KDYNlNUTTlSS21pQmM?e=download)).
This is done in [util.py](/util.py) with a test of windowing in 
[test_util.py](/test_util.py). More tests should be added here.
