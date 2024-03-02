This repository contains code for my experiments with a novel stochastic regularizer similar in principle to dropout and batch norm.

NOTE: This repository is still a work in progress, and there are currently no decisive results. The described regularizer does stabilize the training process, making the network less susceptible to changes in the hyperparameters, but as of now, it does not beat dropout. The intervention is not fully orthogonal to the approach used in dropout, but they could potentially be used together. However, more tests are needed to test this hypothesis.

### Motivation

Regularization plays a crucial role in ensuring good test-time results for a NN. One of the ways regularization can be achieved is by using batch normalization [1] 
(although it is not its main purpose), which adds stochastically to the training process by estimating the mean and standard deviation based on the data in a batch. This process acts as a regularizer because, for each sample, we get a slightly different output in each forward pass. However, the usage of batch norm is limited, and more often than not, layer normalization [2] is used. 

One of the most commonly used regularizers is dropout [3] which randomly selects neurons in the layer and sets their output to 0. This intervention encourages the optimization process to search for subnetworks that can perform the required transformation and then create an ensemble of such subnetworks [4].  

### Intuition

The experiments presented in this repository explore an idea based on the two previously mentioned regularization methods. The basic idea is as follows:

- In a single layer, randomly select k neurons.
- For each selected neuron, generate a scaling factor from the normal distribution N(0, std).
- Scale the output of a given neuron by the given random scaling factor.

Hopefully, this process will encourage the optimization process to generate even more subnetworks, because we randomly strengthen or weaken certain connections in the network (instead of completely blocking them, like in the case of dropout)

The described intervention could also be described as adding stochastic data augmentation to the network structure, when applied to one of the first layers.


### Bibliography
- [1] Ioffe, S. and Szegedy, C., 2015, June. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456). pmlr. - https://arxiv.org/abs/1502.03167
- [2] Ba, J.L., Kiros, J.R. and Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450. - https://arxiv.org/abs/1607.06450
- [3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R., 2014. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), pp.1929-1958. - https://arxiv.org/abs/1502.03167
- [4] Frankle, J. and Carbin, M., 2018. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635. - https://arxiv.org/abs/1803.03635 


