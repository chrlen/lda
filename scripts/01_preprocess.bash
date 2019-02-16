#! /bin/bash

source activate lda
sh scripts/extractSmallSubset.bash
rm -rf dataset/small/*
rm -rf dataset/full/*
ipython preprocess.py dataset/small.xml dataset/small/ &
ipython preprocess.py dataset/simplewiki-20181120-pages-meta-current.xml dataset/large/ &
