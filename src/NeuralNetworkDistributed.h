#ifndef NEURAL_NETWORK_DISTRIBUTED_
#define NEURAL_NETWORK_DISTRIBUTED_

#include "NeuralNetwork.h"
#include "NeuralNetworkParallel.h"
#include "utils/Log.hpp"
#include <mpi.h>
#include <math.h>

class NeuralNetworkDistributed: public NeuralNetwork {
public:
	int argc;
	char** argv;
	NeuralNetwork& nn;

	/**
	 * Creates a dummy neural network to work distributed.
	 *
	 * @param argc The command line arguments count.
	 * @param argv The command line arguments values.
	 * @param nn The neural network to work with.
	 */
	NeuralNetworkDistributed(int argc, char** argv, NeuralNetwork &nn);

	double train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);

	bool saveYAML(std::string const& path);
};

#endif
