#!/usr/bin/env bash

mkdir dataset
cd dataset

#Download data set here, choose "All pages, current versions only"
#https://dumps.wikimedia.org/simplewiki/20181120/
wget https://dumps.wikimedia.org/simplewiki/20181120/simplewiki-20181120-pages-meta-current.xml.bz2
bzip2 -d simplewiki-20181120-pages-meta-current.xml.bz2

sh ../scripts/extractSmallSubset.bash
