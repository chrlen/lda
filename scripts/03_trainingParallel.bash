#! /bin/bash

#source activate lda

iterations=3
topics=20

modeldir=$(date | sed -e "s/ /_/g")_${iterations}_${topics}
mkdir /tmp/models/${modeldir}
mkdir /tmp/models/${modeldir}/small
mkdir /tmp/models/${modeldir}/full
python trainingParallel.py ${iterations} ${topics} dataset/small /tmp/models/${modeldir}/small
#python training.py ${iterations} ${topics} dataset/full models/${modeldir}/full
