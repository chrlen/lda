from lda.dataset import DataSet
from lda.inference import LDA

path = 'dataset/small.xml'
#path = 'dataset/simplewiki-20181120-pages-meta-current.xml'

dataset = DataSet(path=path)
model = LDA()
model.fit(dataset)
