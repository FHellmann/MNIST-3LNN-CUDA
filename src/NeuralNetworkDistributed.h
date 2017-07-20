#include "NeuralNetwork.h"
#include "NeuralNetworkParallel.h"
#include <mpi.h>
#include <math.h>

class NeuralNetworkDistributed : public NeuralNetwork {
public:
	int argc;
	char** argv;
	NeuralNetwork& nn;

	NeuralNetworkDistributed(int argc, char** argv, NeuralNetwork &nn);

	double train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold,
			double const max_derivation);

	bool saveYAML(std::string const& path);
};
