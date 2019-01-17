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


class LDA():
    def __init__(self,
                 maxit=1000,
                 verbose=True,
                 readOutIterations=10
                 ):
        self.verbose = verbose
        self.maxit = maxit

        self.alpha = None
        self.beta = None
        self.iterations = 0
        self.converged = False
        self.readOutIterations = readOutIterations
        self.lastReadOut = 0

    def numOf(self):
        return self.dataset

    def fit(self, dataset,
            nTopics=5):
        self.nTopics = nTopics
        self.dataset = dataset

# -------------------------------- Initialize --------------------------------
        start = time.perf_counter()

        # M: Number of documents
        # K: Number of topics
        # V: number of Terms

        # z_mn : W
        self.topicAssociations_z = hlp.randomMultimatrix(
            dataset.numOfDocuments(),
            dataset.dictionarySize(),
            nTopics)

        # M x K
        self.documentTopic_count_n_m = np.zeros(
            (dataset.numOfDocuments(),
             nTopics)
        )

        # K x v
        self.topicTerm_count_n_k = np.zeros(
            (nTopics,
             dataset.dictionarySize())
        )

        for document in range(dataset.numOfDocuments()):
            for term in range(dataset.dictionarySize()):
                topic = self.topicAssociations_z[document, term]
                self.documentTopic_count_n_m[document, topic] += 1
                self.topicTerm_count_n_k[topic, term] += 1

        # M
        self.documentTopic_sum_n_m = np.sum(
            self.documentTopic_count_n_m, axis=1)
        assert(
            len(self.documentTopic_sum_n_m.shape) == 1
        )
        assert(
            self.documentTopic_sum_n_m.shape[0] == dataset.numOfDocuments()
        )

        # K
        self.topicTerm_sum_n_k = np.sum(self.topicTerm_count_n_k, axis=1)
        assert(
            len(self.topicTerm_sum_n_k.shape) == 1
        )
        assert(
            self.topicTerm_sum_n_k.shape[0] == nTopics
        )

        end = time.perf_counter()
        self.initializazionTime = end - start
        if self.verbose:
            print("LDA => Initialization took: " +
                  "{:10.4f}".format(self.initializazionTime) + "s")

# -------------------------------- Burn-In phase --------------------------
# -------------------------------- Sampling --------------------------------
        if self.verbose:
            print("LDA => fitting to Dataset: " + str(dataset.matrix.shape))

        start = time.perf_counter()

        for iteration in tqdm(range(self.maxit), desc='EM: '):
            # For the current assignment of k to a term t for word w_{m,n}
            # multinomial sampling acc. to Eq. 78 (decrements from previous step)
            # For new assignments of z_{m,n} to the term t for word w_{m,n}

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
            print("LDA => Fitting took: " +
                  "{:10.4f}".format(self.inferenceTime) + "s")


if __name__ == '__main__':
    dataset = DataSet()
    model = LDA()
    model.fit(dataset)
