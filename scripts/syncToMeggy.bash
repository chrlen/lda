#! /bin/bash
rsync -av --delete ./ --exclude .* --exclude lda/.git --exclude venv --exclude tex --exclude dataset --exclude models --exclude modelsGensim --exclude .idea si65dev@meggy.rz.uni-jena.de:git/lda
