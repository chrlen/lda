from lda.inference import LDA
from lda.dataset import DataSet

path = 'dataset/simplewiki-20181120-pages-meta-current.xml'
dataset = DataSet()
dataset.load(path)
dataset.saveToDir('/tmp/dataset/')

loadedDataset = DataSet()
loadedDataset.loadFromDir('/tmp/dataset/')

model = LDA(maxit=500)
model.fitParallel(loadedDataset, nTopics=30)
model.saveToDir('models/small/')
