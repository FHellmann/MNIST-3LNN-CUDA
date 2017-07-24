#ifndef NEURALNETWORKPARALLEL_H_
#define NEURALNETWORKPARALLEL_H_

#include <omp.h>
#include <iostream>
#include "NeuralNetwork.h"
#include "utils/Log.hpp"

class NeuralNetworkParallel: public NeuralNetwork {
public:
	/**
	 * Creates a new simple 3-layer neural network.
	 *
	 * @param inpCount The input layer node count.
	 * @param hidCount the hidden layer node count.
	 * @param outCount The output layer node count.
	 * @param learningRate The learning rate of this neural network.
	 */
	NeuralNetworkParallel(const int inpCount, const int hidCount,
			const int outCount, const double learningRate);

	/**
	 * Copy ctr.
	 */
	NeuralNetworkParallel(NeuralNetworkParallel const&);

	/*
	double train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);
			*/

	void backPropagateOutputLayer(const int targetClassification);

	void backPropagateHiddenLayer(const int targetClassification);

	void updateNodeWeights(const LayerType layertype, const int id,
				double error);

	class LayerParallel: public Layer {
	public:
		LayerParallel(const int nodeCount, const int weightCount,
				const LayerType layerType, const ActFctType actFctType,
				Layer* previous);

		LayerParallel(LayerParallel const&);

		void calcLayer();

		//void calcNodeOutput(Node* node);
	};
};

#endif
