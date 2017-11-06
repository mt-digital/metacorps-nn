'''
Making three or six files, depending on whether 2012 and 2016 or just 2016
'''
import collections
import nltk
import os
import string

from datetime import datetime
from nltk.corpus import stopwords
from pymongo import MongoClient

client = MongoClient()

docs = client.metacorps.iatv_document

stopwords = set(stopwords.words('english'))

for year in [2016]:  # [2012, 2016]:

    for network in ['MSNBCW', 'CNNW', 'FOXNEWSW']:

        subset = docs.find({
            'network': network,
            'start_localtime': {
                '$gte': datetime(year, 9, 1),
                '$lte': datetime(year, 11, 30)
            }
        })

        filename = network + '-' + str(year) + '.txt'

        subset_list = list(subset)
        subset_len = len(subset_list)

        clean_tokens = []
        for idx, doc in enumerate(subset_list):
            a, b, c, d = (idx + 1, subset_len, network, year)
            print('{} out of {} complete for {} {}'.format(a, b, c, d),
                  end='\r')
            # Tokenize, removing punctuation, appostrophes, and
            # commercials. Commercials are in lower-case letters,
            # and thus identified.
            tokens = nltk.word_tokenize(
                doc['document_data']
            )

            # See here for removing stopwords:
            # https://www.google.com/search?q=nltk+stopwords
            clean_tokens += [
                token.lower() for token in tokens
                if token not in string.punctuation
                and not any(c.islower() for c in token)
                and len(token) > 1
                and token.isalpha()
                and token.lower() not in stopwords
            ]

        counts = dict(collections.Counter(clean_tokens).most_common())

        # remove rare words and stopwords
        final_tokens = [
            t for t in clean_tokens
            if counts[t] >= 10
        ]

        open(filename, 'w+').write(' '.join(final_tokens))
