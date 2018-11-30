from lda.dataset import DataSet
from lda.inference import LDA

dataset = DataSet(path='dataset/small.xml')

model = LDA()
model.fit(dataset)