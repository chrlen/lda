#! /bin/bash

source activate lda

python preprocess.py 'dataset/small.xml' "dataset/small/"
python preprocess.py 'dataset/simplewiki-20181120-pages-meta-current.xml' "dataset/large/"
