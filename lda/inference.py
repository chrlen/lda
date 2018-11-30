from lda.dataset import DataSet
import time


class LDA():
    def __init__(self,
                 maxit=1000,
                 verbose=True
                 ):
        self.verbose = verbose


    def fit(self, dataset):
        if self.verbose:
            print("LDA => fitting to Dataset: " + str(dataset.matrix.shape))

        start = time.perf_counter()
        end = time.perf_counter()

        self.inferenceTime = end - start

        if self.verbose:
            print("LDA => Fitting took: " + "{:10.4f}".format(self.inferenceTime) + "s")

if __name__ == '__main__':
    dataset = DataSet()
    model = LDA()
    model.fit(dataset)
