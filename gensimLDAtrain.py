from gensim.models.ldamulticore import LdaMulticore
from lda.dataset import DataSet
import multiprocessing as mp

import scipy.stats as spst
import numpy as np
import numpy.random as npr

from gensim.test.utils import datapath

#path = 'dataset/small.xml'
cores = mp.cpu_count() - 1
path = 'dataset/simplewiki-20181120-pages-meta-current.xml'

dataset = DataSet(path=path)

lda = LdaMulticore(corpus=dataset.documents,
                   num_topics=20,
                   id2word=dataset.dictionary,
                   workers=cores)

temp_file = datapath("/home/me/Desktop/models/largeModel")
lda.save(temp_file)
