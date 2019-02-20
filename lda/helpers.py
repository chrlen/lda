import scipy.stats as spst
import numpy as np
import multiprocessing as mp
import ctypes


def sharedZeros(nRows, nColumns, ctype=ctypes.c_int):
    shared_array_base = mp.Array(ctype, nRows * nColumns)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(nRows, nColumns)
    return(shared_array)


def sharedArray(data, ctype=ctypes.c_int):
    shared_array_base = mp.Array(ctype, data)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    return(shared_array)


def sharedMultiMatrix(nRows, nColumns, nTopics=5, ctype=ctypes.c_int):
    shared_array_base = mp.Array(
        ctype, randomMultilist(nRows * nColumns, nTopics=nTopics))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(nRows, nColumns)
    return(shared_array)


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
