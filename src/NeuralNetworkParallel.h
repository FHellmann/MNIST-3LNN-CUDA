#include "NeuralNetwork.h"
#include <omp.h>
#include <iostream>

class NeuralNetworkParallel : public NeuralNetwork {
public:
	NeuralNetworkParallel(const int inpCount, const int hidCount, const int outCount,
			const double learningRate);

	NeuralNetworkParallel(NeuralNetworkParallel const&);

	bool weightsInitState;

	void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold,
			double const max_derivation);

	void backPropagateOutputLayer(const int targetClassification);

	void backPropagateHiddenLayer(const int targetClassification);

	void updateNodeWeights(const NeuralNetwork::LayerType layertype,
			const int id, double error);

	class LayerParallel : public Layer {
	public:
		LayerParallel(const int nodeCount, const int weightCount,
				const LayerType layerType, const ActFctType actFctType,
				Layer* previous);

		LayerParallel(LayerParallel const&);

		void calcLayer();

		void calcNodeOutput(Node* node);
	};
};
