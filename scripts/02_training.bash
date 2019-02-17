#! /bin/bash

source activate lda
ipython training.py 1000 50 dataset/small models/small &
ipython training.py 1000 50 dataset/large models/full &
