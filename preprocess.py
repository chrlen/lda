from lda.dataset import DataSet
import sys
import multiprocessing as mp

dataset = DataSet()
dataset.load(sys.argv[1])
dataset.saveToDir(sys.argv[2])
