#! /bin/bash

ipython training.py 10 30 dataset/small models/small &
ipython training.py 10 30 dataset/large models/full &
