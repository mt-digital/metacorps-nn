'''
Export script to create tabular dataset from the metacorps web app's
MongoDB database. A mongodump of this database is available at
http://metacorps.io/static/data/nov-15-2017-metacorps-dump.zip (594M)
'''
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from pymongo import MongoClient


CLIENT = MongoClient()


def make_csv(project_name='Viomet Sep-Nov 2012',
             output_path='augmented_2012.csv',
             random_seed=42):
    '''
    Keep random_seed=42 to reproduce results in paper.
    '''

    project = CLIENT.metacorps.project.find(
        {'name': 'Viomet Sep-Nov 2012'}
    ).next()

    facets = project['facets']

    facet_docs = [
        CLIENT.metacorps.facet.find({'_id': id_}).next()
        # the first three facets are hit, attack, and beat
        for id_ in facets[:3]
    ]

    tokenizer = RegexpTokenizer(r'\w+')

    def _preprocess(text):
        text = text.lower()
        text = ' '.join(tokenizer.tokenize(text))
        return _replacements(text)

    text_metaphor_rows = [
        (
            doc['word'],
            instance['reference_url'],
            _preprocess(instance['text']),
            int(instance['figurative'])
        )
        for doc in facet_docs
        for instance in doc['instances']
    ]

    # create tabular format of data
    data = pd.DataFrame(
        data=text_metaphor_rows,
        columns=['word', 'reference_url', 'text', 'is_metaphor']
    )

    # now sample with replacement to get a balanced dataset
    n_metaphor = data['is_metaphor'].sum()
    n_rows = len(data)
    n_to_sample = n_rows - (2 * n_metaphor)
    print(
        (
            '{} out of {} are metaphor. '
            'Adding {} more rows by sampling with replacement for a total '
            'number of {} rows.'
        ).format(n_metaphor, n_rows, n_to_sample, n_to_sample + n_rows)
    )

    metaphor_rows = data[data.is_metaphor == 1]

    np.random.seed()
    indexes_to_sample = np.random.choice(range(n_metaphor), n_to_sample)

    augmentation = metaphor_rows.iloc[indexes_to_sample]

    data = data.append(augmentation, ignore_index=True)

    data.to_csv(output_path, index=False)

    return data


def _replacements(text):
    text = text.replace('i m ', 'im ')
    text = text.replace('dont t ', 'dont ')
    text = text.replace('u s ', 'U.S. ')
    text = text.replace('it s ', 'it\'s ')
    text = text.replace('we re ', 'we\'re ')
    text = text.replace('they re ', 'they\'re ')
    text = text.replace('can t ', 'can\'t ')
    text = text.replace('palins', 'palin')
    return text
