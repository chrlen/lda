#! /bin/bash
source activate lda
sh scripts/extractSmallSubset.bash
python preprocess.py dataset/small.xml dataset/small/
#python preprocess.py dataset/simplewiki-20181120-pages-meta-current.xml dataset/full/
