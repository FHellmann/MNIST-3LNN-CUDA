# MNIST-3LNN-CUDA
3-Layer Neural Network for MNIST Handwriting Recognition with CUDA

Based on the blog [Simple 3-Layer Neural Network for MNIST Handwriting Recognition](https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) of Matt Lind.

## Dependencies
Depends on
- OMP (gomp 1.0.0)
- MPI 20.0.1
- OpenCV 2.4.9

## Build
To build the project it is necessary to initialize the submodule (yaml-cpp). Afterwards you can build the 
trainer (Debug/Release) - to train your network and safe it in a file - and the query (Debug/Release) - 
to run your safed network on real data. 

### yaml-cpp
1. Init the submodules:
```
git submodule init
git submodule update
```
2. Build `yaml-cpp` as described in `yaml-cpp/install.txt`
   Note that it's sufficient to only build the project. Installing it is not necessary.

## Run
For an overview over the available training options, run:
```
trainLindNet -h
```

### Distributed
To run the trainer in distributed mode (-d or --distributed) it is mendetory to start the program 
with [mpirun](https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php).

## Realisiation
The Realisiation is based on the blog [Simple 3-Layer Neural Network for MNIST Handwriting Recognition](https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) of Matt Lind.

### Parallel
There are different Parallelization Strategies\[4\] how to implement a parallel Neural Network.
- Session Parallelism - Run independent neural network with different initial weights to find in one instance the best solution.
- Exemplar Parallelism - Split the training data into subsets and train each network with a subset. After each training finished, merge the results.
- Node Parallelism - The neurons of the Neural Network were handled by several workers/threads.
- Weight Parallelism - The synapse input is calculated parallel for each node.

We decided to use the Exemplar Parallelism due to the low I/O which is needed to communicate between all threads. The implementation is based on the Parallel Static Gradient Descent Algorithm\[3\] (Algorithm 2).

### CUDA
The CUDA version tries to speed up the neural network training by parallelizing matrix operations and processing multiple training samples in one weight update step. If the feed-forward and back-propagation steps can be expressed by matrix operations, they can be performed efficiently by a GPU. Also applying the non-linear activation function is only done component-wise which benefits from the highly parallel nature of the GPU.

The output and errors of to the same set of weights can be computed simultaneously by multiplying the weight matrices onto inputs/outputs matrices which contain one training example in each column.

The whole approach is summarized in this ![image](presentation/images/cudaConcept.svg).

### Distributed
Distribution in Neural Networks can be effort with Model Parallelism or Data Parallelism or a combination of both \[2\].
![Model Paralellism and Data Parallelism](http://engineering.skymind.io/hubfs/EN_Blog_Post_Images/Distributed_Deep_Learning,_Part_1_An_Introduction_to_Distributed_Training_of_Neural_Networks/ModelDataParallelism.svg?t=1498750359042)
![Combination of Model and Data Parallelism](http://engineering.skymind.io/hubfs/EN_Blog_Post_Images/Distributed_Deep_Learning,_Part_1_An_Introduction_to_Distributed_Training_of_Neural_Networks/ModelAndDataParallelism.svg?t=1498750359042)

Due to the low network I/O we decided to use the Data Parallelism. The machines are most of the time independend from each other. To synchronize the parameters every weight of each node of each machine has to be merged. Therefor a Parameter Averaging or the Asynchronous Stochastic Gradient Descent algorithm can be used. The Parameter Averaging is not as exact as the Stochastic Gradient Descent algorithm. However, this leads us to use the Stochastic Gradient Descent algorithm.

## Performance-Analysis
To evaluate the performance of the Neural Network it was trained with the MNIST-Dataset (60000 Images). 
Every network was trained with 784 Input-Nodes, 20 Hidden-Nodes, 10 Output-Nodes and a learning rate of 0.2. 
The training needs to reach a error lower then n% (=> >(100 - n)% correct recognition) or it should cancel if 
a error derivation of 0.005 occures. It was distributed on 10 machines with 1 master and 9 slaves.

| Distributed | Type        | >93% *  | >94% *  | >95% *  | 
| :---------: | :---------- | :-----: | :-----: | :-----: |
|             | Sequentiell | 9.01534 | 15.7818 | 33.7251 |
|             | Parallel    | 8.92147 | 15.8313 | 23.4899 |
|             | CUDA        |         |         |         |
| X           | Sequentiell | 2.04115 | 2.66057 | 2.85507 |
| X           | Parallel    | 8.04285 | 9.80803 | 12.9221 |
| X           | CUDA        |         |         |         |

\* Correct classification on training dataset.


## Literature
1. [High Performance Parallel Stochastic Gradient Descent in Shared Memory](http://www.ece.ubc.ca/~matei/papers/ipdps16.pdf)
2. [Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks](http://engineering.skymind.io/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks)
3. [Parallelized Stochastic Gradient Descent](http://martin.zinkevich.org/publications/nips2010.pdf)
4. [Parallelization of a Backpropagation Neural Network on a Cluster Computer](http://www.cs.otago.ac.nz/staffpriv/hzy/papers/pdcs03.pdf)
