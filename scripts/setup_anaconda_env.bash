#! /bin/bash

conda create -y --name lda pandas scipy tqdm gensim python=3.7
source activate lda
conda install -y -c anaconda gensim pattern
#conda install -y pandas scipy tqdm
#conda install -y -c conda-forge pyldavis
