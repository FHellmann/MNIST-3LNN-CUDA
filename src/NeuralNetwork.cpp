#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const int inpCount, const int hidCount,
		const int outCount, const double learningRate) :
		learningRate(learningRate) {
	layers.push_back(new Layer(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(
			new Layer(hidCount, inpCount, HIDDEN, SIGMOID,
					layers.back()));
	layers.push_back(
			new Layer(outCount, hidCount, OUTPUT, SIGMOID,
					layers.back()));

	for (int l = 0; l < layers.size() - 1; l++) { // leave out the output layer
		Layer* layer = layers.at(l);
		for (int i = 0; i < layer->nodes.size(); i++) {
			Layer::Node *node = layer->getNode(i);

			for (int j = 0; j < node->weights.size(); j++) {
				node->weights[j] = rand() / (double) (RAND_MAX);
			}

			node->bias = rand() / (double) (RAND_MAX);
		}
	}
}

NeuralNetwork::NeuralNetwork(Layer* inpLayer, Layer* hidLayer, Layer* outLayer,
		double learningRate) :
		learningRate(learningRate) {
	layers.push_back(inpLayer);
	layers.push_back(hidLayer);
	layers.push_back(outLayer);
}

void NeuralNetwork::feedForward() {
	getLayer(HIDDEN)->calcLayer();
	getLayer(OUTPUT)->calcLayer();
}

void NeuralNetwork::backPropagate(const int targetClassification) {
	backPropagateOutputLayer(targetClassification);
	backPropagateHiddenLayer(targetClassification);
}

int NeuralNetwork::getNetworkClassification() {
	Layer *layer = getLayer(OUTPUT);

	double maxOut = 0;
	int maxInd = 0;

	for (int i = 0; i < layer->nodes.size(); i++) {
		Layer::Node *on = layer->getNode(i);

		if (on->output > maxOut) {
			maxOut = on->output;
			maxInd = i;
		}
	}

	return maxInd;
}

NeuralNetwork::Layer* NeuralNetwork::getLayer(
		const LayerType layerType) {
	for (int i = 0; i < layers.size(); i++) {
		Layer *layer = layers.at(i);
		if (layer->layerType == layerType)
			return layer;
	}
	return nullptr;
}

void NeuralNetwork::backPropagateOutputLayer(const int targetClassification) {
	Layer *layer = getLayer(OUTPUT);

	for (int i = 0; i < layer->nodes.size(); i++) {
		Layer::Node *node = layer->getNode(i);

		int targetOutput = (i == targetClassification) ? 1 : 0;

		double errorDelta = targetOutput - node->output;
		double errorSignal = errorDelta
				* layer->getActFctDerivative(node->output);

		updateNodeWeights(OUTPUT, i, errorSignal);
	}
}

void NeuralNetwork::backPropagateHiddenLayer(const int targetClassification) {
	Layer *ol = getLayer(OUTPUT);
	Layer *layer_hidden = getLayer(HIDDEN);

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

void NeuralNetwork::updateNodeWeights(const NeuralNetwork::LayerType layertype,
		const int id, double error) {
	Layer *layer = getLayer(layertype);
	Layer::Node *node = layer->getNode(id);
	Layer *prevLayer = layer->previousLayer;

	for (int i = 0; i < node->weights.size(); i++) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		node->weights.at(1) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
}

NeuralNetwork::Layer::Layer(const int nodeCount, const int weightCount,
		const LayerType layerType, const ActFctType actFctType, Layer* previous) :
		layerType(layerType), actFctType(actFctType), previousLayer(previous) {
	for (int i = 0; i < nodeCount; i++)
		nodes.push_back(new Node(weightCount));
}

NeuralNetwork::Layer::Node* NeuralNetwork::Layer::getNode(int index) {
	return nodes.at(index);
}

void NeuralNetwork::Layer::calcLayer() {
	for (int i = 0; i < nodes.size(); i++) {
		Node *node = getNode(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

void NeuralNetwork::Layer::calcNodeOutput(Node* node) {

	// Start by adding the bias
	node->output = node->bias;

	for (int i = 0; i < previousLayer->nodes.size(); i++) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}

void NeuralNetwork::Layer::activateNode(Node* node) {
	switch (actFctType) {
	case SIGMOID: {
		node->output = 1 / (1 + (exp((double) -node->output)));
		break;
	}
	case TANH: {
		node->output = tanh(node->output);
		break;
	}
	}
}

double NeuralNetwork::Layer::getActFctDerivative(double outVal) {
	double dVal = 0;
	switch (actFctType) {
	case SIGMOID: {
		dVal = outVal * (1 - outVal);
		break;
	}
	case TANH: {
		dVal = 1 - pow(tanh(outVal), 2);
		break;
	}
	}
	return dVal;
}

NeuralNetwork::Layer::Node::Node(const int weightCount) :
		Node(weightCount, 0, 0) {
}

NeuralNetwork::Layer::Node::Node(const int weightCount, const double bias,
		const double output) :
		bias(bias), output(output) {
	for (int i = 0; i < weightCount; i++)
		weights.push_back(0);
}

