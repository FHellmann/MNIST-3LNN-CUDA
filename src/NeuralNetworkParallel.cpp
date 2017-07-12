#include "NeuralNetworkParallel.h"

using namespace std;

NeuralNetworkParallel::NeuralNetworkParallel(const int inpCount, const int hidCount,
		const int outCount, const double _learningRate) {

	learningRate = _learningRate;
	layers.push_back(new LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(new LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(new LayerParallel(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

	//#pragma omp parallel for
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

	weightsInitState = true;
}

NeuralNetworkParallel::NeuralNetworkParallel(NeuralNetworkParallel const& net) :
		NeuralNetwork(net.learningRate) {

	weightsInitState = net.weightsInitState;

	layers.reserve(net.layers.size());
	for (size_t i = 0; i < net.layers.size(); ++i) {
		// Make a deep copy of the layers
		layers.push_back(new LayerParallel(*dynamic_cast<LayerParallel*>(net.layers[i])));
	}

	//getLayer(HIDDEN)->previousLayer = getLayer(INPUT);
	//getLayer(OUTPUT)->previousLayer = getLayer(HIDDEN);

	// And set the previous layer in the new network.
	for (size_t i = 0; i < layers.size(); ++i) {
		size_t prevLayerIdx = 0;
		for (; prevLayerIdx < net.layers.size(); ++prevLayerIdx) {
			if (net.layers[prevLayerIdx] == net.layers[i]->previousLayer) {
				break;
			}
		}
		layers[i]->previousLayer = layers[prevLayerIdx];
	}
}

void mergeNeuralNetworks(NeuralNetworkParallel& omp_in, NeuralNetworkParallel& omp_out) {
	omp_out.weightsInitState = omp_in.weightsInitState;
	for(int l=0; l < omp_in.layers.size(); l++) {
		NeuralNetwork::Layer* layerIn = omp_in.layers.at(l);
		NeuralNetwork::Layer* layerOut = omp_out.layers.at(l);
		for(int n=0; n < layerIn->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* nodeIn = layerIn->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeOut = layerOut->nodes.at(n);
			for(int w=0; w < nodeIn->weights.size(); w++) {
				double weightIn = nodeIn->weights.at(w);
				double weightOutOld = nodeOut->weights.at(w);
				nodeOut->weights.at(w) += weightIn - weightOutOld;
			}
		}
	}
}

void resetWeights(NeuralNetworkParallel& net) {
	for(int l=0; l < net.layers.size(); l++) {
		NeuralNetwork::Layer* layer = net.layers.at(l);
		for(int n=0; n < layer->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* node = layer->nodes.at(n);
			for(int w=0; w < node->weights.size(); w++) {
				node->weights.at(w) = 0;
			}
		}
	}
}

#pragma omp declare reduction(mergeWeights:NeuralNetworkParallel:mergeNeuralNetworks(omp_in, omp_out))

void NeuralNetworkParallel::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {

	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();
	while (needsFurtherTraining) {

		size_t errCount = 0;
		int every_ten_percent = images.size() / 10;

		// Loop through all images in the file
		if(weightsInitState) {
			resetWeights(*this);
		}

		#pragma omp parallel shared(errCount)
		{
			NeuralNetworkParallel nnp_copy(*this);

			#pragma omp for
			for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {
				// Convert the MNIST image to a standardized vector format and feed into the network
				nnp_copy.feedInput(images[imgCount]);

				// Feed forward all layers (from input to hidden to output) calculating all nodes' output
				nnp_copy.feedForward();

				// Back propagate the error and adjust weights in all layers accordingly
				nnp_copy.backPropagate(labels[imgCount]);

				// Classify image by choosing output cell with highest output
				int classification = nnp_copy.getNetworkClassification();
				if (classification != labels[imgCount])
					errCount++;

				nnp_copy.weightsInitState = false;

				// Display progress during training
				if ((imgCount % every_ten_percent) == 0) {
					cout << "x";
					cout.flush();
				}
			}

			// merge network weights together
			#pragma omp critical
			mergeNeuralNetworks(nnp_copy, *this);
		}

		double newError = static_cast<double>(errCount) / static_cast<double>(images.size());
		if (newError < error) {
			error = newError;
		} else if (newError > error + max_derivation) {
			// The error increases again. This is not good.
			needsFurtherTraining = false;
		}

		if (error < training_error_threshold) {
			needsFurtherTraining = false;
		}

		cout << " Error: " << error * 100.0 << "%" << endl;
	}

	cout << endl;
}

void NeuralNetworkParallel::backPropagateOutputLayer(const int targetClassification) {
	Layer *layer = getLayer(OUTPUT);

	//#pragma omp parallel for
	for (int i = 0; i < layer->nodes.size(); i++) {
		Layer::Node *node = layer->getNode(i);

		int targetOutput = (i == targetClassification) ? 1 : 0;

		double errorDelta = targetOutput - node->output;
		double errorSignal = errorDelta
				* layer->getActFctDerivative(node->output);

		updateNodeWeights(OUTPUT, i, errorSignal);
	}
}

void NeuralNetworkParallel::backPropagateHiddenLayer(const int targetClassification) {
	Layer *ol = getLayer(OUTPUT);
	Layer *layer_hidden = getLayer(HIDDEN);

	//#pragma omp parallel for
	for (int h = 0; h < layer_hidden->nodes.size(); h++) {
		Layer::Node *hn = layer_hidden->getNode(h);

		double outputcellerrorsum = 0;

		//#pragma omp parallel for reduction(+:outputcellerrorsum)
		for (int o = 0; o < ol->nodes.size(); o++) {

			Layer::Node *on = ol->getNode(o);

			int targetOutput = (o == targetClassification) ? 1 : 0;

			double errorDelta = targetOutput - on->output;
			double errorSignal = errorDelta
					* ol->getActFctDerivative(on->output);

			outputcellerrorsum += errorSignal * on->weights[h];
		}

		double hiddenErrorSignal = outputcellerrorsum
				* layer_hidden->getActFctDerivative(hn->output);

		updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	}
}

void NeuralNetworkParallel::updateNodeWeights(const NeuralNetwork::LayerType layertype,
		const int id, double error) {
	Layer *layer = getLayer(layertype);
	Layer::Node *node = layer->getNode(id);
	Layer *prevLayer = layer->previousLayer;

	//#pragma omp parallel for
	for (int i = 0; i < node->weights.size(); i++) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		if(weightsInitState)
			node->weights.at(i) = 0;
		node->weights.at(i) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
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

void NeuralNetworkParallel::LayerParallel::calcLayer() {
	//#pragma omp parallel for
	for (int i = 0; i < nodes.size(); i++) {
		Node *node = getNode(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

void NeuralNetworkParallel::LayerParallel::calcNodeOutput(Node* node) {
	// Start by adding the bias
	node->output = node->bias;

	//#pragma omp parallel for
	for (int i = 0; i < previousLayer->nodes.size(); i++) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}
