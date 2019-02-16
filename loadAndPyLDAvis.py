import pickle
import pyLDAvis
path = 'models/small/'

phi = pickle.load(open(path + 'phi.pickle', 'rb'))
theta = pickle.load(open(path + 'theta.pickle', 'rb'))
docLengths = pickle.load(open(path + 'docLengths.pickle', 'rb'))
vocabulary = pickle.load(open(path + 'vocabulary.pickle', 'rb'))
termFrequencys = pickle.load(open(path + 'termFrequencys.pickle', 'rb'))

prep = pyLDAvis.prepare(
    phi,
    theta,
    docLengths,
    vocabulary,
    termFrequencys
)

pyLDAvis.display(prep)
