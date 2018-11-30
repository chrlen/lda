import numpy as np
import pytest


@pytest.fixture()
def simpleModel():
    import lda.generativeModel as gm
    topicWordConcentration_beta = np.array([1, 1, 1, 1, 1, 1, 1])
    documentTopicConcentration_alpha = np.array([1, 1, 1])
    return gm.GenMod(mDocuments=100,
                     kTopics=documentTopicConcentration_alpha.shape[0],
                     topicWordConcentration_beta=topicWordConcentration_beta,
                     documentTopicConcentration_alpha=documentTopicConcentration_alpha,
                     poissonMoment=500)


def test_init(simpleModel):
    assert 1 == 1
