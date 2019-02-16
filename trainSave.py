from lda.inference import LDA
from lda.dataset import DataSet


path = 'dataset/small.xml'
#path = 'dataset/simplewiki-20181120-pages-meta-current.xml'

dataset = DataSet()
dataset.loadFromDir('dataset/small/')
model = LDA(maxit=50)
model.fit(dataset)
