#! /bin/bash

#source activate lda

iterations=5000
topics=20

modeldir=$(date | sed -e "s/ /_/g")_${iterations}_${topics}
mkdir models/${modeldir}
mkdir models/${modeldir}/small
mkdir models/${modeldir}/full
#python trainingParallel.py ${iterations} ${topics} dataset/small models/${modeldir}/small
python trainingParallel.py ${iterations} ${topics} dataset/full models/${modeldir}/full
