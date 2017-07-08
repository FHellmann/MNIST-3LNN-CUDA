#include "NeuralNetwork.cpp"

using namespace std;

class NeuralNetworkParallel : public NeuralNetwork {
	void feedInput(cv::Mat const& image) {
	
		Layer* inputLayer = getLayer(INPUT);
		size_t const numPixels = image.cols * image.rows;
	
		size_t const loopCount = min(numPixels, inputLayer->nodes.size());
		cv::MatConstIterator_<uint8_t> it = image.begin<uint8_t>();
		#pragma omp parallel for
		for (int i = 0; i < loopCount; ++i, ++it) {
			inputLayer->nodes[i]->output = static_cast<double>(*it);
		}
	}
	
	void feedForward() {
		getLayer(HIDDEN)->calcLayer();
		getLayer(OUTPUT)->calcLayer();
	}
	
	void backPropagate(const int targetClassification) {
		backPropagateOutputLayer(targetClassification);
		backPropagateHiddenLayer(targetClassification);
	}
	
	void backPropagateOutputLayer(const int targetClassification) {
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
	
	void backPropagateHiddenLayer(const int targetClassification) {
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
	
	void updateNodeWeights(const NeuralNetwork::LayerType layertype,
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
	
	void Layer::calcLayer() {
		for (int i = 0; i < nodes.size(); i++) {
			Node *node = getNode(i);
			calcNodeOutput(node);
			activateNode(node);
		}
	}
	
	void Layer::calcNodeOutput(Node* node) {
	
		// Start by adding the bias
		node->output = node->bias;
	
		for (int i = 0; i < previousLayer->nodes.size(); i++) {
			Node *prevLayerNode = previousLayer->getNode(i);
			node->output += prevLayerNode->output * node->weights.at(i);
		}
	}
	
	void Layer::activateNode(Node* node) {
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
};
