import scipy.stats as spst
import numpy as np


def getIndex(arr):
    counter = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            return i
        else:
            counter = counter + 1


def randomMultilist(length, nTopics=5):
    return list(map(getIndex, spst.multinomial(1, [1 / nTopics] * nTopics).rvs(length)))


def randomMultimatrix(nColumns=10, nRows=12, nTopics=5):
    columns = range(nColumns)
    return np.matrix(
        list(
            map(lambda x: list(map(getIndex, spst.multinomial(
                1, [1 / nTopics] * nTopics).rvs(nRows))), columns)
        )
    )

    # def increment(
    #        topicAssociations_z,
    #        mat,
    #        row, col):
    #    lock = th.Lock()
    #    index = topicAssociations_z[row, col]
    #    lock.acquire(mat[row, col])
    #    mat[row, index] += 1
    #    index = mat[row, col]
    #    lock.release()

    # with mp.Pool(4) as p:
    #    p.map(lambda x: increment(x[0], x[1],
    #                              self.documentTopic_count_n_m), indices())
