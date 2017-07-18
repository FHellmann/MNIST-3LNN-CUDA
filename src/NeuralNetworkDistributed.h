#include "NeuralNetworkParallel.h"
#include <mpi.h>

class NeuralNetworkDistributed : public NeuralNetwork {
public:
	NeuralNetworkDistributed(const int inpCount, const int hidCount, const int outCount,
			const double learningRate);

	void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold,
			double const max_derivation);
};
