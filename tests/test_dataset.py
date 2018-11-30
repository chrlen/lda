import numpy as np
import pytest
import os

@pytest.fixture()
def simpleModel():
    import lda.dataset as dts
    return dts.DataSet()

def test_init(simpleModel):
    print(simpleModel.docs)
    assert 1 == 1
