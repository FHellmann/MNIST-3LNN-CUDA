#include "NeuralNetwork.h"
#include "NeuralNetworkParallel.h"
#include <mpi.h>

class NeuralNetworkDistributed : public NeuralNetwork {
public:
	int world_size;
	int curr_rank;

	NeuralNetworkDistributed(const int _worldSize, const int _currRank,
			const int inpCount, const int hidCount, const int outCount,
			const double learningRate);

	void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold,
			double const max_derivation);
};
