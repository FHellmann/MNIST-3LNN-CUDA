#include "NeuralNetworkParallel.h"

using namespace std;

NeuralNetworkParallel::NeuralNetworkParallel(const int inpCount, const int hidCount,
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

NeuralNetworkParallel::NeuralNetworkParallel(NeuralNetworkParallel const& net) :
		NeuralNetwork(net.learningRate) {
	layers.reserve(net.layers.size());
	for (size_t i = 0; i < net.layers.size(); ++i) {
		// Make a deep copy of the layers
		layers.push_back(new LayerParallel(*dynamic_cast<LayerParallel*>(net.layers[i])));
		if(i > 0)
			layers.at(i)->previousLayer = layers.at(i - 1);
	}

	//getLayer(HIDDEN)->previousLayer = getLayer(INPUT);
	//getLayer(OUTPUT)->previousLayer = getLayer(HIDDEN);

	// And set the previous layer in the new network.
	/*
	for (size_t i = 0; i < layers.size(); ++i) {
		size_t prevLayerIdx = 0;
		for (; prevLayerIdx < net.layers.size(); ++prevLayerIdx) {
			if (net.layers[prevLayerIdx] == net.layers[i]->previousLayer) {
				break;
			}
		}
		layers[i]->previousLayer = layers[prevLayerIdx];
	}
	*/
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
			nodeOut->bias += nodeIn->bias - nodeReset->bias;
		}
	}
}

void NeuralNetworkParallel::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {

	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();
	while (needsFurtherTraining) {

		double newError = 0;
		int every_ten_percent = images.size() / 10;

		NeuralNetworkParallel nnp_merge(*this);

		#pragma omp parallel default(shared)
		{
			NeuralNetworkParallel nnp_local(*this);
			size_t localErrCount = 0;

			#pragma omp for
			for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {
				// Convert the MNIST image to a standardized vector format and feed into the network
				nnp_local.feedInput(images[imgCount]);

				// Feed forward all layers (from input to hidden to output) calculating all nodes' output
				nnp_local.feedForward();

				// Back propagate the error and adjust weights in all layers accordingly
				nnp_local.backPropagate(labels[imgCount]);

				// Classify image by choosing output cell with highest output
				int classification = nnp_local.getNetworkClassification();
				if (classification != labels[imgCount])
					localErrCount++;

				// Display progress during training
				if ((imgCount % every_ten_percent) == 0) {
					cout << "x";
					cout.flush();
				}
			}

			//#pragma omp critical
			newError += static_cast<double>(localErrCount) / static_cast<double>(images.size());

			// merge network weights together
			#pragma omp critical
			mergeNeuralNetworks(nnp_local, nnp_merge, this);

			//#pragma omp critical
			//cout << "Thread-" << omp_get_thread_num() << ": Error=" << localErrCount << ", Images=" << localImageProcessed << endl;
		}

		mergeNeuralNetworks(nnp_merge, *this, this);

		if (newError < error) {
			error = newError;
		}

		needsFurtherTraining = !(error < training_error_threshold
				|| newError > error + max_derivation);

		cout << " Error: " << error * 100.0 << "%, NewError: " << newError * 100.0 << "%" << endl;
	}

	cout << endl;
}

NeuralNetworkParallel::LayerParallel::LayerParallel(const int nodeCount, const int weightCount,
		const LayerType _layerType, const ActFctType _actFctType, Layer* _previous) :
			Layer(nodeCount, weightCount, _layerType, _actFctType, _previous) {
}

NeuralNetworkParallel::LayerParallel::LayerParallel(LayerParallel const& layer) :
		Layer(layer.layerType, layer.actFctType) {

	nodes.reserve(layer.nodes.size());
	for (LayerParallel::Node* node : layer.nodes) {
		nodes.push_back(new Node(*node));
	}
}
