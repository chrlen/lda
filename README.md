# Latent Dirichlet Allocation

Implement variational inference algorithm for latent dirichlet allocation.
Train model on a small subset of wikipedia.
Evaluate and visualize with pyLDAvis

To reproduce check the following scripts:
- scripts/setup_anaconda_env.bash to build suitable anaconda-environment.
- scripts/00_setup.bash to download the wikipedia dataset.
- scripts/extractSmallSubset.bash to extract a subset of the dataset.
- scripts/01_preprocess.bash to process xml files and save the dictionary and wordcounts for each document.
- scripts/02_training.bash to estimate the distribution parameters and save the
- to visualize run the jupyter-notebook with the same name and point it to the location of your trained model (by setting the path in the second cell). A Small model is in

There are three relevant Python classes in the package **lda**.
- Dataset in lda/dataset.py for all corpus preprocessing operations as well as loading and saving datasets in the native Python serialization format pickle.
- LDA in lda/inference.py to perform the inference algorithm on a dataset
- GenMod in lda/generativeModel.py to sample from a LDA model given the hyperparameters
