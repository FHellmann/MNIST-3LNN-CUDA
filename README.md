# MNIST-3LNN-CUDA
3-Layer Neural Network for MNIST Handwriting Recognition with CUDA

Based on the blog [Simple 3-Layer Neural Network for MNIST Handwriting Recognition](https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) of Matt Lind.

## Dependencies
Depends on
- OpenCV 2
- TCLAP

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
The Neural Network can be trained with the following options:
```
USAGE: 

   trainer-Debug/trainLindNet  [--derivation <maxDerivation>]
                               [--trainingError <trainingError>] [-l
                               <learningRate>] [--outputNodes
                               <outputLayerNodes>] [--hiddenNodes
                               <hiddenLayerNodes>] [--inputNodes
                               <inputLayerNodes>] [-d] [-t <type>] [-n
                               <path>] --mnist <path> [--] [--version]
                               [-h]


Where: 

   --derivation <maxDerivation>
     The max derivation between to erros after two train samples where
     proceed.

   --trainingError <trainingError>
     The training error when the neural network should quit work if the
     value is reached.

   -l <learningRate>,  --learningRate <learningRate>
     The learning rate of the neural network.

   --outputNodes <outputLayerNodes>
     The amount of output nodes.

   --hiddenNodes <hiddenLayerNodes>
     The amount of hidden nodes.

   --inputNodes <inputLayerNodes>
     The amount of input nodes.

   -d,  --distributed
     The neural network will be executed distributed.

   -t <type>,  --networkType <type>
     The neural network type (sequentiell, parallel, cuda).

   -n <path>,  --lindNet <path>
     yaml file for saving the resulting net.

   --mnist <path>
     (required)  Folder containing the MNIST files.

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.


   Train the LindNet with the MNIST database.

```

### Distributed
To run the trainer in distributed mode (-d or --distributed) it is mendetory to start the program 
with [mpirun](https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php).

## Performance-Analysis
To evaluate the performance of the Neural Network it was trained with the MNIST-Dataset (60000 Images). 
Every network was trained with 784 Input-Nodes, 20 Hidden-Nodes, 10 Output-Nodes and a learning rate of 0.2. 
The training needs to reach a error lower then 4% (=> >96% correct recognition) or it should cancel if 
a error derivation of 0.005 occures.

### Sequentiell

### Parallel

### CUDA

### Parallel - Distributed

### CUDA - Distributed

## Literature
[High Performance Parallel Stochastic Gradient Descent in Shared Memory](http://www.ece.ubc.ca/~matei/papers/ipdps16.pdf)
[Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks](http://engineering.skymind.io/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks)
[Parallelized Stochastic Gradient Descent](http://martin.zinkevich.org/publications/nips2010.pdf)
[Parallelization of a Backpropagation Neural Network on a Cluster Computer](http://www.cs.otago.ac.nz/staffpriv/hzy/papers/pdcs03.pdf)