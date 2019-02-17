#! /bin/bash

source activate lda

iterations=500
topics=50

modeldir=$(date | sed -e "s/ /_/g")_${iterations}_${topics}
mkdir models/${modeldir}
mkdir models/${modeldir}/small
mkdir models/${modeldir}/full
python training.py ${iterations} ${topics} dataset/small models/${modeldir}/small &
python training.py ${iterations} ${topics} dataset/full models/${modeldir}/full &
