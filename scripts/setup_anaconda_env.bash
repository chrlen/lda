#! /bin/bash

conda create -y --name lda python=3.7
source activate lda
conda install -y pandas scipy gensim pattern tqdm
conda install -y -c conda-forge pyldavis
