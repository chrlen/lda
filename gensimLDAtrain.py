from gensim.models.ldamulticore import LdaMulticore
import multiprocessing as mp

from gensim.test.utils import datapath
import pickle

#path = 'dataset/small.xml'
cores = mp.cpu_count() - 1
datasetPath = 'dataset/full/'

dictionary = pickle.load(open(datasetPath + "dictionary.pickle", 'rb'))
corpus = pickle.load(open(datasetPath + "corpus.pickle", 'rb'))

lda = LdaMulticore(corpus=corpus,
                   num_topics=50,
                   id2word=dictionary,
                   workers=cores)

#temp_file = datapath('home/si65dev/git/lda/models/smallModel')
lda.save('models/full/full50')
