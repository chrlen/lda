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
from functools import partial
import json
import multiprocessing as mp
import pickle


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

    def fitParallel(self, dataset,
                    nTopics=5,
                    nCores=mp.cpu_count() - 1):
        self.nTopics = nTopics
        self.dataset = dataset
        if self.alpha == None:
            self.alpha = np.repeat(50 / nTopics, nTopics)
        if self.beta == None:
            self.beta = np.repeat(0.01, dataset.dictionarySize())

        if self.verbose:
            print("LDA-Model => fitting to dataset")
        start = time.perf_counter()

        global alpha
        global beta
        alpha = self.alpha
        beta = self.beta

        global documents
        documents = dataset.documents

        # M: Number of documents
        # K: Number of topics
        # V: number of Terms

        global topicAssociations_z
        topicAssociations_z = hlp.sharedMultiMatrix(
            dataset.numOfDocuments(), np.max(dataset.documentLengths()), nTopics
        )

        # M x K
        global documentTopic_count_n_mk
        documentTopic_count_n_mk = hlp.sharedZeros(
            dataset.numOfDocuments(),
            nTopics
        )

        # K x v
        global topicTerm_count_n_kt
        topicTerm_count_n_kt = hlp.sharedZeros(
            nTopics,
            dataset.dictionarySize()
        )

        for documentIndex in range(dataset.numOfDocuments()):
            document = dataset.documents[documentIndex]
            wordIndex = 0
            for pair in document:
                termIndex = pair[0]
                for c in range(pair[1]):
                    topicIndex = topicAssociations_z[documentIndex][wordIndex]
                    documentTopic_count_n_mk[documentIndex,
                                             topicIndex] += 1
                    topicTerm_count_n_kt[topicIndex, termIndex] += 1
                    wordIndex += 1

        # M
        global documentTopic_sum_n_m
        documentTopic_sum_n_m = np.sum(
            documentTopic_count_n_mk, axis=1)
        documentTopic_sum_n_m = hlp.sharedArray(
            documentTopic_sum_n_m)
        assert (
            len(documentTopic_sum_n_m.shape) == 1
        )
        assert (
            documentTopic_sum_n_m.shape[0] == dataset.numOfDocuments()
        )

        # K
        global topicTerm_sum_n_k
        topicTerm_sum_n_k = np.sum(topicTerm_count_n_kt, axis=1)
        topicTerm_sum_n_k = hlp.sharedArray(topicTerm_sum_n_k)
        assert (
            len(topicTerm_sum_n_k.shape) == 1
        )
        assert (
            topicTerm_sum_n_k.shape[0] == nTopics
        )

        # end = time.perf_counter()
        end = time.perf_counter()
        self.initializazionTime = end - start
        if self.verbose:
            print("LDA => Initialization took: {:10.4f}".format(
                self.initializazionTime) + "s")

        # -------------------------------- Sampling --------------------------------
        if self.verbose:
            print("LDA => fitting to Dataset")

        start = time.perf_counter()

        global processDocument

        def processDocument(
            documentIndex,
            documents=documents,
            topicAssociations_z=topicAssociations_z,
            documentTopic_count_n_mk=documentTopic_count_n_mk,
            topicTerm_count_n_kt=topicTerm_count_n_kt,
            documentTopic_sum_n_m=documentTopic_sum_n_m,
            topicTerm_sum_n_k=topicTerm_sum_n_k,
            beta=beta,
            alpha=alpha,
            nTopics=nTopics
        ):
            document = documents[documentIndex]
            wordIndex = 0
            for pair in document:
                termIndex = pair[0]
                for c in range(pair[1]):
                    previousTopicIndex = topicAssociations_z[documentIndex][wordIndex]

                    # For the current assignment of k to a term t for word w_{m,n}
                    documentTopic_count_n_mk[documentIndex,
                                             previousTopicIndex] -= 1
                    documentTopic_sum_n_m[documentIndex] -= 1
                    topicTerm_count_n_kt[previousTopicIndex,
                                         termIndex] -= 1
                    topicTerm_sum_n_k[previousTopicIndex] -= 1

                    # multinomial sampling acc. to Eq. 78 (decrements from previous step)

                    params = np.zeros(nTopics)
                    for topicIndex in range(nTopics):
                        n = topicTerm_count_n_kt[topicIndex,
                                                 termIndex] + beta[termIndex]
                        d = topicTerm_sum_n_k[topicIndex] + \
                            beta[termIndex]
                        f = documentTopic_count_n_mk[documentIndex,
                                                     topicIndex] + alpha[topicIndex]
                        params[topicIndex] = (n / d) * f

                    # Scale
                    params = np.asarray(params).astype('float64')
                    params = params / np.sum(params)

                    # if np.sum(params) < 1:
                    #    newTopicIndex = hlp.getIndex(
                    #        spst.multinomial(1, params).rvs()[0])
                    # else:
                    newTopicIndex = previousTopicIndex

                    topicAssociations_z[documentIndex][wordIndex] = newTopicIndex
                    # For new assignments of z_{m,n} to the term t for word w_{m,n}
                    documentTopic_count_n_mk[documentIndex,
                                             newTopicIndex] += 1
                    documentTopic_sum_n_m[documentIndex] += 1
                    topicTerm_count_n_kt[newTopicIndex,
                                         termIndex] += 1
                    topicTerm_sum_n_k[newTopicIndex] += 1
                    wordIndex += 1

        for iteration in tqdm(range(self.maxit), desc='Sampling: '):
            with mp.Pool(mp.cpu_count() - 1) as p:
                p.map(processDocument, range(len(dataset.documents)))

            self.iterations += 1

            if self.converged and self.lastReadOut > self.readOutIterations:
                print("reading")

            if self.iterations > self.maxit:
                self.converged = True
                if self.verbose:
                    print("LDA.fit() => Maximum number of iterations reached!")

        self.topicAssociations_z = topicAssociations_z
        self.documentTopic_count_n_mk = documentTopic_count_n_mk
        self.topicTerm_count_n_kt = topicTerm_count_n_kt
        self.documentTopic_sum_n_m = documentTopic_sum_n_m
        self.topicTerm_sum_n_k = topicTerm_sum_n_k

        self.compute_phi()
        self.compute_theta()

        end = time.perf_counter()

        self.inferenceTime = end - start

        if self.verbose:
            print("LDA => Fitting took: {:10.4f}".format(
                self.inferenceTime) + "s")
            print("LDA => Convergence took: {:10.4f}".format(self.iterations))

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

        partialMultilist = partial(hlp.randomMultilist, nTopics=nTopics)
        self.topicAssociations_z = list(
            map(partialMultilist, dataset.documentLengths()))

        # M x K
        self.documentTopic_count_n_mk = np.zeros(
            (dataset.numOfDocuments(),
             nTopics)
        )

        print("Dsize:", dataset.dictionarySize())
        # K x v
        self.topicTerm_count_n_kt = np.zeros(
            (nTopics,
             dataset.dictionarySize())
        )

        for documentIndex in range(dataset.numOfDocuments()):
            document = dataset.documents[documentIndex]
            wordIndex = 0
            for pair in document:
                termIndex = pair[0]
                for c in range(pair[1]):
                    topicIndex = self.topicAssociations_z[documentIndex][wordIndex]
                    self.documentTopic_count_n_mk[documentIndex,
                                                  topicIndex] += 1
                    self.topicTerm_count_n_kt[topicIndex, termIndex] += 1
                    wordIndex += 1

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
        # end = time.perf_counter()
        end = time.perf_counter()
        self.initializazionTime = end - start
        if self.verbose:
            print("LDA => Initialization took: {:10.4f}".format(
                self.initializazionTime) + "s")

        # -------------------------------- Sampling --------------------------------
        if self.verbose:
            print("LDA => fitting to Dataset")

        start = time.perf_counter()

        for iteration in tqdm(range(self.maxit), desc='Sampling: '):
            for documentIndex in range(len(dataset.documents)):
                document = dataset.documents[documentIndex]
                wordIndex = 0
                for pair in document:
                    termIndex = pair[0]
                    for c in range(pair[1]):
                        previousTopicIndex = self.topicAssociations_z[documentIndex][wordIndex]

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
                        params = np.asarray(params).astype('float64')
                        params = params / np.sum(params)
                        newTopicIndex = hlp.getIndex(
                            spst.multinomial(1, params).rvs()[0])

                        self.topicAssociations_z[documentIndex][wordIndex] = newTopicIndex
                        # For new assignments of z_{m,n} to the term t for word w_{m,n}
                        self.documentTopic_count_n_mk[documentIndex,
                                                      newTopicIndex] += 1
                        self.documentTopic_sum_n_m[documentIndex] += 1
                        self.topicTerm_count_n_kt[newTopicIndex,
                                                  termIndex] += 1
                        self.topicTerm_sum_n_k[newTopicIndex] += 1
                        wordIndex += 1

            self.iterations += 1

            if self.converged and self.lastReadOut > self.readOutIterations:
                print("reading")

            if self.iterations > self.maxit:
                self.converged = True
                if self.verbose:
                    print("LDA.fit() => Maximum number of iterations reached!")

        self.compute_phi()
        self.compute_theta()

        end = time.perf_counter()

        self.inferenceTime = end - start

        if self.verbose:
            print("LDA => Fitting took: {:10.4f}".format(
                self.inferenceTime) + "s")
            print("LDA => Convergence took: {:10.4f}".format(self.iterations))

    def saveToDir(self, savePath, protocol=2):
        # Safe topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
        with open(savePath + 'phi.pickle', 'wb') as handle:
            pickle.dump(
                self.phi, handle,
                protocol=protocol
            )

        with open(savePath + 'theta.pickle', 'wb') as handle:
            pickle.dump(
                self.theta, handle,
                protocol=protocol
            )

        with open(savePath + 'docLengths.pickle', 'wb') as handle:
            pickle.dump(
                self.dataset.docLengths, handle,
                protocol=protocol
            )

        with open(savePath + 'vocabulary.pickle', 'wb') as handle:
            pickle.dump(
                list(map(
                    lambda x: self.dataset.dictionary[x], self.dataset.dictionary.keys())), handle,
                protocol=protocol
            )
        with open(savePath + 'termFrequencys.pickle', 'wb') as handle:
            pickle.dump(self.dataset.termCounts, handle,
                        protocol=protocol
                        )

    def compute_phi(self):
        """Calculate Parameters of The topic-term multinomial"""
        self.phi = np.zeros((self.nTopics, self.dataset.dictionarySize()))
        for topicIndex, termIndex in tqdm(it.product(range(self.nTopics), range(self.dataset.dictionarySize())), desc='Computing Phi'):
            self.phi[topicIndex, termIndex] = (self.topicTerm_count_n_kt[topicIndex, termIndex]
                                               + self.beta[termIndex]) / (self.topicTerm_sum_n_k[topicIndex] + self.beta[termIndex])

    def compute_theta(self):
        """Calculate Parameters of The document-topic multinomial"""
        self.theta = np.zeros((self.dataset.numOfDocuments(), self.nTopics))
        for documentIndex, topicIndex in tqdm(it.product(range(self.dataset.numOfDocuments()), range(self.nTopics)), desc='Computing Theta'):
            self.theta[documentIndex, topicIndex] = (self.documentTopic_count_n_mk[documentIndex, topicIndex] + self.alpha[topicIndex]) / (
                self.documentTopic_sum_n_m[documentIndex] + self.alpha[topicIndex])


if __name__ == '__main__':
    dataset = DataSet()
    model = LDA()
    model.fit(dataset)
