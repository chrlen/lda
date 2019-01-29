from gensim.models.ldamulticore import LdaMulticore
from lda.dataset import DataSet
import multiprocessing as mp

import scipy.stats as spst
import numpy as np
import numpy.random as npr

import pickle

from gensim.test.utils import datapath

path = 'dataset/small.xml'
cores = mp.cpu_count() - 1
path = 'dataset/simplewiki-20181120-pages-meta-current.xml'
#savePath = "/home/me/Desktop/simpleWiki/"

dataset = DataSet(path=path)

with open(savePath + 'corpus.pickle', 'wb') as handle:
    pickle.dump(dataset.documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(savePath + 'dictionary.pickle', 'wb') as handle:
    pickle.dump(dataset.dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
