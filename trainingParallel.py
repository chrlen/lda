from lda.dataset import DataSet
from lda.inference import LDA
import sys

iterations = int(sys.argv[1])
topics = int(sys.argv[2])

dataset = DataSet()
dataset.loadFromDir(sys.argv[3] + '/')
model = LDA(maxit=iterations)
model.fitParallel(dataset, nTopics=topics)
model.saveToDir(sys.argv[4] + '/')
