from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim
import pickle

datasetPath='dataset/full/'

lda = LdaMulticore.load("models/full/full50")
dictionary = pickle.load(open(datasetPath + "dictionary.pickle", 'rb'))
corpus = pickle.load(open(datasetPath + "corpus.pickle", 'rb'))

prep = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
pyLDAvis.show(prep)

