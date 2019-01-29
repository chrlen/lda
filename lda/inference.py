from lda.dataset import DataSet
from tqdm import tqdm
import time
import scipy.sparse as sps
import scipy as sp
import numpy as np
import numpy.random as npr
import scipy.stats as spst
import lda.helpers as hlp
import multiprocessing as mp
import itertools as it
import threading as th
import json


class LDA():
    def __init__(self,
                 maxit=3,
                 verbose=True,
                 readOutIterations=10,
                 estimateHyperparameters=True,
                 # Mixture proportions; length = num of topics
                 alpha=None,
                 # Mixture components ; length = num of terms
                 beta=None
                 ):
        self.verbose = verbose
        self.maxit = maxit

        self.alpha = None
        self.beta = None
        self.iterations = 0
        self.converged = False
        self.readOutIterations = readOutIterations
        self.lastReadOut = 0

        if self.verbose:
            print("LDA-Model => constructed")

        def saveJson(self, file):
        saveDict = {'topic_term_dists': self.phi,
                    'doc_topic_dists': self.topicTerm_count_n_kt,
                    'doc_lengths': dataset.documentLengths(),
                    'vocab': dataset,
                    'term_frequency': data_input['term.frequency']}
        jsonString = json.dumps(my_dictionary)

    def fit(self, dataset,
            nTopics=5):
        self.nTopics = nTopics
        self.dataset = dataset
        if self.alpha == None:
            self.alpha = np.repeat(50 / nTopics, nTopics)
        if self.beta == None:
            self.beta = np.repeat(0.01, dataset.dictionarySize())

        if self.verbose:
            print("LDA-Model => fitting to dataset")
        start = time.perf_counter()

        # M: Number of documents
        # K: Number of topics
        # V: number of Terms

        # z_mn : W
        self.topicAssociations_z = hlp.randomMultimatrix(
            dataset.numOfDocuments(),
            np.max(list(map(len, dataset.documents))),
            nTopics)

        # M x K
        self.documentTopic_count_n_mk = np.zeros(
            (dataset.numOfDocuments(),
             nTopics)
        )

        # K x v
        self.topicTerm_count_n_kt = np.zeros(
            (nTopics,
             dataset.dictionarySize())
        )

        for documentIndex in range(dataset.numOfDocuments()):
            document = dataset.documents[documentIndex]
            for wordIndex in range(len(document)):
                word = document[wordIndex]
                termIndex = dataset.dictionary.token2id[word]
                topicIndex = self.topicAssociations_z[documentIndex, wordIndex]
                self.documentTopic_count_n_mk[documentIndex, topicIndex] += 1
                self.topicTerm_count_n_kt[topicIndex, termIndex] += 1

        # M
        self.documentTopic_sum_n_m = np.sum(
            self.documentTopic_count_n_mk, axis=1)
        assert (
            len(self.documentTopic_sum_n_m.shape) == 1
        )
        assert (
            self.documentTopic_sum_n_m.shape[0] == dataset.numOfDocuments()
        )

        # K
        self.topicTerm_sum_n_k = np.sum(self.topicTerm_count_n_kt, axis=1)
        assert (
            len(self.topicTerm_sum_n_k.shape) == 1
        )
        assert (
            self.topicTerm_sum_n_k.shape[0] == nTopics
        )
        end = time.perf_counter()
        self.initializazionTime = end - start
        if self.verbose:
            print("LDA => Initialization took: {:10.4f}".format(
                self.initializazionTime) + "s")

        # -------------------------------- Sampling --------------------------------
        if self.verbose:
            print("LDA => fitting to Dataset")

        start = time.perf_counter()

        for iteration in tqdm(range(self.maxit), desc='Gibb's: '):
            for documentIndex in range(len(dataset.documents)):
                document = dataset.documents[documentIndex]
                for wordIndex in range(len(document)):
                    word = document[wordIndex]
                    termIndex = dataset.dictionary.token2id[word]
                    previousTopicIndex = self.topicAssociations_z[documentIndex, wordIndex]

                    # For the current assignment of k to a term t for word w_{m,n}
                    self.documentTopic_count_n_mk[documentIndex,
                                                  previousTopicIndex] -= 1
                    self.documentTopic_sum_n_m[documentIndex] -= 1
                    self.topicTerm_count_n_kt[previousTopicIndex,
                                              termIndex] -= 1
                    self.topicTerm_sum_n_k[previousTopicIndex] -= 1

                    # multinomial sampling acc. to Eq. 78 (decrements from previous step)

                    params = np.zeros(self.nTopics)
                    for topicIndex in range(self.nTopics):
                        n = self.topicTerm_count_n_kt[topicIndex,
                                                      termIndex] + self.beta[termIndex]
                        d = self.topicTerm_sum_n_k[topicIndex] + \
                            self.beta[termIndex]
                        f = self.documentTopic_count_n_mk[documentIndex,
                                                          topicIndex] + self.alpha[topicIndex]
                        params[topicIndex] = (n / d) * f

                    # Scale
                    #params = np.abs(params)
                    params = np.asarray(params).astype('float64')
                    params = params / np.sum(params)
                    newTopicIndex = hlp.getIndex(
                        spst.multinomial(1, params).rvs()[0])

                    self.topicAssociations_z[documentIndex,
                                             wordIndex] = newTopicIndex
                    # For new assignments of z_{m,n} to the term t for word w_{m,n}
                    self.documentTopic_count_n_mk[documentIndex,
                                                  newTopicIndex] += 1
                    self.documentTopic_sum_n_m[documentIndex] += 1
                    self.topicTerm_count_n_kt[newTopicIndex, termIndex] += 1
                    self.topicTerm_sum_n_k[newTopicIndex] += 1

            self.iterations += 1
            # if self.verbose:
            #    print("LDA.fit() => iteration: " + str(self.iterations))

            if self.converged and self.lastReadOut > self.readOutIterations:
                print("reading")

            if self.iterations > self.maxit:
                self.converged = True
                if self.verbose:
                    print("LDA.fit() => Maximum number of iterations reached!")

        end = time.perf_counter()

        self.inferenceTime = end - start

        if self.verbose:
            print("LDA => Fitting took: {:10.4f}".format(
                self.inferenceTime) + "s")
            print("LDA => Convergence took: {:10.4f}".format(self.iterations))


if __name__ == '__main__':
    dataset = DataSet()
    model = LDA()
    model.fit(dataset)
