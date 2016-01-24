# LSTM-Autoencoder

This project implements the LSTM Autoencoder for sequence modeling.
The model reads a sequence and decodes itself. 
The model can be easily extended for any encoder-decoder task.

## Dependencies
This code requires [Torch7](http://torch.ch/) and [nngraph](http://github.com/torch/nngraph)

## Datasets
In general, with proper parameter settings the model can recover 80%-90% of the words, when tested on a small subset of Toronto movie book corpus[http://www.cs.toronto.edu/~mbweb/].

## Usage
To train a model with default setting, simply run
    th LSTMAutoencoder.lua
The code generates samples at validation time, to inspect the effective of reconstruction.
One may consider to use the Autoencoder to obtain general purpose sentence vectors, or as a pretraining step for downstream tasks 

## References
* Li, Jiwei, Minh-Thang Luong, and Dan Jurafsky,
  "[A hierarchical neural autoencoder for paragraphs and documents](http://arxiv.org/abs/1506.01057)",
  *arXiv preprint arXiv:1506.01057 (2015)*.

* Dai, Andrew M., and Quoc V. Le,
  "[Semi-supervised sequence learning](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning.pdf)",
  *Advances in Neural Information Processing Systems. 2015*.
