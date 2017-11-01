'''
Making three or six files, depending on whether 2012 and 2016 or just 2016
'''
import nltk
import string

from datetime import datetime
from pymongo import MongoClient

client = MongoClient()

docs = client.metacorps.iatv_document

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

        with open(filename, 'w+') as f:
            for doc in subset:
                # Tokenize, removing punctuation, appostrophes, and
                # commercials. Commercials are in lower-case letters,
                # and thus identified.
                tokens = nltk.word_tokenize(
                    doc['document_data'].replace("'", " ")
                )

                clean_tokens = [token for token in tokens
                                if token not in string.punctuation
                                and not any(c.islower() for c in token)]

                f.write(' '.join(clean_tokens))
