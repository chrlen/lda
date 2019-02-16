from lda.inference import LDA
from lda.dataset import DataSet


#xpath = 'dataset/small.xml'
path = 'dataset/simplewiki-20181120-pages-meta-current.xml'
dataset = DataSet()
dataset.loadFromDir(path)
model = LDA(maxit=3)
model.fit(dataset, nTopics=30)
model.saveToDir('models/small/')
