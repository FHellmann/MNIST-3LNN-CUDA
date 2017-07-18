#include "NeuralNetworkDistributed.h"

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(const int inpCount, const int hidCount,
		const int outCount, const double _learningRate) {
	learningRate = _learningRate;
	layers.push_back(new LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(new LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(new LayerParallel(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

	for (int l = 0; l < layers.size() - 1; l++) { // leave out the output layer
		Layer* layer = layers.at(l);
		for (int i = 0; i < layer->nodes.size(); i++) {
			Layer::Node *node = layer->getNode(i);

			for (int j = 0; j < node->weights.size(); j++) {
				node->weights[j] = 0.7 * (rand() / (double) (RAND_MAX));
				if(j % 2)
					node->weights[j] = -node->weights[j]; // make half of the weights negative
			}

			node->bias = rand() / (double) (RAND_MAX);
			if(i % 2)
				node->bias = -node->bias; // make half of the bias weights negative
		}
	}
}

void mergeNeuralNetworks(NeuralNetworkParallel& omp_in, NeuralNetworkParallel& omp_out, NeuralNetworkParallel* reset) {
	for(int l=0; l < omp_in.layers.size(); l++) {
		NeuralNetwork::Layer* layerIn = omp_in.layers.at(l);
		NeuralNetwork::Layer* layerOut = omp_out.layers.at(l);
		NeuralNetwork::Layer* layerReset = reset->layers.at(l);
		for(int n=0; n < layerIn->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* nodeIn = layerIn->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeOut = layerOut->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeReset = layerReset->nodes.at(n);
			for(int w=0; w < nodeIn->weights.size(); w++) {
				nodeOut->weights.at(w) += nodeIn->weights.at(w) - nodeReset->weights.at(w);
			}
			//nodeOut->bias += nodeIn->bias - nodeReset->bias;
		}
	}
}

void NeuralNetworkDistributed::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_Size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {
    	// --- MASTER ---

    	// 1. Send (MPI_Scatter) training set to slaves

    	// 2. Send (MPI_Bcast) neural network structure to slaves

    	// 3. Receive (MPI_Gather) initialization information from slaves

    	// 4. Send (MPI_Bcast) weights to slaves

    	// 5. Receive (MPI_Gather) all delta weights from slaves

    	// 6. Check whether stop (go on to 7.) or repeat (back to 4.)

    	// 7. Merge into this neural network
    } else {
    	// --- SLAVE ---

    	// 1. Receive (MPI_Scatter) training set

    	// 2. Receive (MPI_Bcast) neural network structure

    	// 3. Send (MPI_Gather) initialization finished

    	// 4. Receive (MPI_Bcast) weights

    	// 5. Perform training on training set

    	// 6. Send (MPI_Gather) delta weight
    }

	MPI_Finalize();
}
