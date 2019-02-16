import numpy.random as npr
import numpy as np
import scipy.stats as spst
from lda.inference import LDA
from lda.dataset import DataSet


#corputPath = 'dataset/small.xml'
#path = 'dataset/simplewiki-20181120-pages-meta-current.xml'

dataset = DataSet()
dataset.load(path)
model = LDA(maxit=10)
model.fit(dataset)
