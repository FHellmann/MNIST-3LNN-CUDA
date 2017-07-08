#include "NeuralNetworkParallel.h"

NeuralNetworkParallel::NeuralNetworkParallel(const int inpCount, const int hidCount,
		const int outCount, const double learningRate) :
		learningRate(learningRate) {
	layers.push_back(new LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(new LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(new LayerParallel(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

	#pragma omp parallel for
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

void NeuralNetworkParallel::feedInput(cv::Mat const& image) {
	Layer* inputLayer = getLayer(INPUT);
	size_t const numPixels = image.cols * image.rows;

	size_t const loopCount = min(numPixels, inputLayer->nodes.size());
	cv::MatConstIterator_<uint8_t> it = image.begin<uint8_t>();
	#pragma omp parallel for
	for (int i = 0; i < loopCount; ++i, ++it) {
		inputLayer->nodes[i]->output = static_cast<double>(*it);
	}
}

void NeuralNetworkParallel::feedForward() {
	//#pragma omp parallel
	{
		getLayer(HIDDEN)->calcLayer();
		getLayer(OUTPUT)->calcLayer();
	}
}

void NeuralNetworkParallel::backPropagate(const int targetClassification) {
	//#pragma omp parallel
	{
		backPropagateOutputLayer(targetClassification);
		backPropagateHiddenLayer(targetClassification);
	}
}

void NeuralNetworkParallel::backPropagateOutputLayer(const int targetClassification) {
	Layer *layer = getLayer(OUTPUT);
	
	#pragma omp parallel for
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

	#pragma omp parallel for
	for (int h = 0; h < layer_hidden->nodes.size(); h++) {
		Layer::Node *hn = layer_hidden->getNode(h);

		double outputcellerrorsum = 0;

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

	#pragma omp parallel for
	for (int i = 0; i < node->weights.size(); i++) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		node->weights.at(1) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
}

NeuralNetworkParallel::LayerParallel::LayerParallel(const int nodeCount, const int weightCount,
		const LayerType layerType, const ActFctType actFctType, Layer* previous) :
		layerType(layerType), actFctType(actFctType), previousLayer(previous) {
	for (int i = 0; i < nodeCount; i++)
		nodes.push_back(new Node(weightCount));
}

void NeuralNetworkParallel::LayerParallel::calcLayer() {
	#pragma omp parallel for
	for (int i = 0; i < nodes.size(); i++) {
		Node *node = getNode(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

void NeuralNetworkParallel::LayerParallel::calcNodeOutput(Node* node) {
	// Start by adding the bias
	node->output = node->bias;

	#pragma omp parallel for
	for (int i = 0; i < previousLayer->nodes.size(); i++) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}
