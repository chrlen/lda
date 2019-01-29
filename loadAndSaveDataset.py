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
savePath = "~/Desktop/dataset"

dataset = DataSet(path=path)
