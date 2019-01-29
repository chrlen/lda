import xml.etree.ElementTree as ET
import multiprocessing as mp
import gensim as gsm
import scipy.sparse as sps
import scipy as sp
import numpy as np
from functools import partial
import time
from tqdm import tqdm


def preprocessText(page, onlyOverview=True):
    # Shortcut to get it running
    # return page
    return gsm.parsing.preprocessing.preprocess_string(page)


def tuples2Matrix(tuples,
                  dictionarySize,
                  matrixFormat,
                  matrixDType):
    matrix = matrixFormat(
        (1, dictionarySize),
        dtype=matrixDType
    )

    for t in tuples:
        matrix[0, t[0]] = t[1]

    return matrix


class DataSet:
    def __init__(self,
                 path='../dataset/small.xml',
                 sparse=True,
                 verbose=True,
                 matrixFormat='sparse',
                 matrixDType=sp.int8):
        self.verbose = verbose
        self.path = path
        if (matrixFormat == 'sparse'):
            self.matrixFormat = sps.csc_matrix
        else:
            self.matrixFormat = np.matrix
        self.matrixDType = matrixDType

        if verbose:
            print("Dataset => Loading File")

        start = time.perf_counter()
        documents, self.documentLengths, self.dictionary = self.loadXMLFile()
        end = time.perf_counter()
        self.loadTime = end - start

        if verbose:
            print("Dataset => Parsing " + str(len(documents)) +
                  " documents took: " + "{:10.4f}".format(self.loadTime) + "s")

        if verbose:
            print("Dataset => Building Matrix")

        start = time.perf_counter()
        self.documents = self.buildMatrix(documents, self.dictionary)
        end = time.perf_counter()
        self.buildMatrixTime = end - start

        if verbose:
            print("Dataset => Building took: " +
                  "{:10.4f}".format(self.buildMatrixTime) + "s")

        if verbose:
            print("Dataset => Constructed")

    def numOfDocuments(self):
        return len(self.documents)

    def documentLengths(self):
        return self.documentLengths

    def dictionarySize(self):
        return len(self.dictionary)

    def buildMatrix(self, documents, dictionary):
        partialTuples2Matrix = partial(tuples2Matrix,
                                       dictionarySize=self.dictionarySize(),
                                       matrixFormat=self.matrixFormat,
                                       matrixDType=self.matrixDType)

        with mp.Pool(mp.cpu_count() - 1) as p:
            counts = p.map(dictionary.doc2bow, tqdm(
                documents, desc='Counting words'))
            # rows = p.map(partialTuples2Matrix, tqdm(
            #    counts, desc='Generating sparse document vectors'))

        return counts

    def loadXMLFile(self):
        documents = list()
        root = ET.parse(self.path).getroot()
        xmlNamespaces = {'root': 'http://www.mediawiki.org/xml/export-0.10/'}

        # Extract text-attribute of pages in Wikipedia-namespace '0'
        texts = [
            page.find('root:revision', xmlNamespaces)
            .find('root:text', xmlNamespaces).text
            for page in root.findall('root:page', xmlNamespaces)
            if 0 == int(page.find('root:ns', xmlNamespaces).text)
        ]

        if self.verbose:
            print('Parse xml')

        # Parallel preprocessing of pages
        with mp.Pool(mp.cpu_count() - 1) as p:
            documents = p.map(preprocessText, tqdm(
                texts, desc='Preprocessing text'))
        documentsLengths = list(map(len, documents))
        # Build gensim dictionary
        dictionary = gsm.corpora.dictionary.Dictionary(documents)
        return [documents, documentsLengths, dictionary]


if __name__ == '__main__':
    dataset = DataSet()
    print(dataset.numOfDocuments())
    print(dataset.dictionarySize())
    print(dataset.matrix.T.dot(dataset.matrix))
