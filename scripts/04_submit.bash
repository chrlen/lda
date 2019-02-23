#! /bin/bash

cd tex
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
rm *.aux *.log

cd ..

tar --exclude='.git' --exclude='*.pyc' --exclude='venv' --exclude='dataset' --exclude='modelsGensim' --exclude='models' --exclude='.ipynb_checkpoints' --exclude='.pytest_cache' --exclude='.pytest_cache' --exclude='.idea' -zcvf ../christian_lengert_153767_lda.tar.gz tex/main.pdf ./
