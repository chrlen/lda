from lda.dataset import DataSet
import time


class LDA():
    def __init__(self,
                 maxit=1000,
                 verbose=True
                 ):
        self.verbose = verbose
        self.maxit = maxit

        self.alpha = None
        self.beta = None
        self.iterations = 0
        self.converged = False

    def expectation(self):
        """ For each document, find the optimizing values of the variational parameters """
        print("Expect")

    def maximization(self):
        """ Maximize the resulting lower bound on the log likelihood """
        print("Maximize")

    def numOfDocuments():
        return this.dataset.numOfDocuments()

    def num

    def fit(self, dataset):
        if self.verbose:
            print("LDA => fitting to Dataset: " + str(dataset.matrix.shape))

        start = time.perf_counter()

        while not self.converged:
            self.expectation()
            self.maximization()
            self.iterations += 1
            if self.verbose:
                print("LDA.fit() => iteration: " + str(self.iterations))
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
