#include "NeuralNetwork.h"
#include <omp.h>

using namespace std;

class NeuralNetworkParallel : public NeuralNetwork {
public:
	NeuralNetworkParallel(const int inpCount, const int hidCount, const int outCount,
			const double learningRate);

	void backPropagateOutputLayer(const int targetClassification);

	void backPropagateHiddenLayer(const int targetClassification);

	void updateNodeWeights(const NeuralNetwork::LayerType layertype,
			const int id, double error);

	class LayerParallel : public Layer {
	public:
		LayerParallel(const int nodeCount, const int weightCount,
				const LayerType layerType, const ActFctType actFctType,
				Layer* previous);

		void calcLayer();

		void calcNodeOutput(Node* node);
	};
};
