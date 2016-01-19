# LSTM-Autoencoder

This project implements the LSTM Autoencoder (Li et al., 2015) for sequence modeling.
The model reads a sequence and decodes itself. 
The model can be easily extended for any encoder-decoder task.

## Dependencies
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph)

## Datasets
In general, with proper parameter settings the model can recover 80%-90% of the words, when tested on a small subset of Toronto movie book corpus[http://www.cs.toronto.edu/~mbweb/].

## Usage
To train a model with default setting, simply run
    th LSTMAutoencoder.lua --gpuid 0
The code generates samples at validation time, to inspect the effective of reconstruction.
