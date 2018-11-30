import numpy as np

from scipy.stats import poisson
from scipy.stats import multinomial
from scipy.stats import dirichlet


class GenMod:
    def __init__(self,
                 mDocuments,
                 kTopics,
                 topicWordConcentration_beta,
                 documentTopicConcentration_alpha,
                 poissonMoment=500):
        self.mDocuments = mDocuments
        self.kTopics = kTopics
        self.topicWordDirichlet = dirichlet(topicWordConcentration_beta)
        self.topicWordMultinomialsPhi = self.topicWordDirichlet.rvs(size=self.kTopics)
        self.documentTopicDir = dirichlet(documentTopicConcentration_alpha)
        self.documentTopicMultinomialsTheta = self.documentTopicDir.rvs(mDocuments)
        self.poissonMoment = poissonMoment
        self.docs, self.wordTopicLists = self.randomize()

    def randomize(self):
        docs = list()
        wordTopicLists = list()
        for documentIndex in range(self.mDocuments):
            documentLengths = [poisson.rvs(self.poissonMoment)
                               for i in range(self.mDocuments)]
            doc = list()
            docs.append(doc)

            wordTopicList = list()
            wordTopicLists.append(wordTopicList)

            for wordIndex in range(documentLengths[documentIndex]):
                topicIndex = multinomial.rvs(
                    p=self.documentTopicMultinomialsTheta[documentIndex, :], n=1)
                wordTopicList.append(topicIndex)
                word = multinomial.rvs(
                    p=self.topicWordMultinomialsPhi[topicIndex, :][0], n=1)
                doc.append(word)
        return ([docs, wordTopicLists])
