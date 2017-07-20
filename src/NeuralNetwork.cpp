#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork() :
		learningRate(0.0) {

}

NeuralNetwork::NeuralNetwork(NeuralNetwork const& net) :
		learningRate(net.learningRate) {

	layers.reserve(net.layers.size());
	for (Layer* layer : net.layers) {
		// Make a deep copy of the layers
		layers.push_back(new Layer(*layer));
	}

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

NeuralNetwork::NeuralNetwork(const int inpCount, const int hidCount,
		const int outCount, const double learningRate) :
		learningRate(learningRate) {
	layers.push_back(new Layer(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(
			new Layer(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(
			new Layer(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

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

NeuralNetwork::NeuralNetwork(double const rate) :
		learningRate(rate) {
}

NeuralNetwork::~NeuralNetwork() {
	for (Layer* layer : layers) {
		delete layer;
	}
	layers.clear();
}

void NeuralNetwork::feedInput(cv::Mat const& image) {

	Layer* inputLayer = getLayer(INPUT);
	size_t const numPixels = image.cols * image.rows;

	size_t const loopCount = min(numPixels, inputLayer->nodes.size());
	cv::MatConstIterator_<uint8_t> it = image.begin<uint8_t>();
	for (int i = 0; i < loopCount; ++i, ++it) {
		inputLayer->nodes[i]->output = ((*it > 128.0) ? 1.0 : 0.0);
	}
}

double NeuralNetwork::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {
	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();
	while (needsFurtherTraining) {

		int every_ten_percent = images.size() / 10;
		size_t errCount = 0;
		// Loop through all images in the file
		for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {

			// Convert the MNIST image to a standardized vector format and feed into the network
			feedInput(images[imgCount]);

			// Feed forward all layers (from input to hidden to output) calculating all nodes' output
			feedForward();

			// Back propagate the error and adjust weights in all layers accordingly
			backPropagate(labels[imgCount]);

			// Classify image by choosing output cell with highest output
			int classification = getNetworkClassification();
			if (classification != labels[imgCount])
				errCount++;

			// Display progress during training
			//if ((imgCount % every_ten_percent) == 0)
			//	cout << "x"; cout.flush();
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

		//cout << " Error: " << error * 100.0 << "%" << endl;
	}

	//cout << endl;

	return error;
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

NeuralNetwork::Layer* NeuralNetwork::getLayer(const LayerType layerType) {
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

		int const targetOutput = (i == targetClassification) ? 1 : 0;

		double const errorDelta = targetOutput - node->output;
		double const errorSignal = errorDelta * layer->getActFctDerivative(node->output);

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

			int const targetOutput = (o == targetClassification) ? 1 : 0;

			double const errorDelta = targetOutput - on->output;
			double const errorSignal = errorDelta * ol->getActFctDerivative(on->output);

			outputcellerrorsum += errorSignal * on->weights[h];
		}

		double const hiddenErrorSignal = outputcellerrorsum * layer_hidden->getActFctDerivative(hn->output);

		updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	}
}

void NeuralNetwork::updateNodeWeights(const NeuralNetwork::LayerType layertype,
		const int id, double error) {
	Layer *layer = getLayer(layertype);
	Layer::Node *node = layer->getNode(id);
	Layer *prevLayer = layer->previousLayer;

	for (size_t i = 0; i < node->weights.size(); ++i) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		node->weights.at(i) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
}

NeuralNetwork::Layer::Layer(const int nodeCount, const int weightCount,
		const LayerType layerType, const ActFctType actFctType, Layer* previous) :
		layerType(layerType), actFctType(actFctType), previousLayer(previous) {
	for (int i = 0; i < nodeCount; i++)
		nodes.push_back(new Node(weightCount));
}

NeuralNetwork::Layer::Layer(const LayerType layerType,
		const ActFctType actFctType, Layer* previous) :
		Layer(0, 0, layerType, actFctType, previous) {
}

NeuralNetwork::Layer::Layer(Layer const& layer) :
	layerType(layer.layerType),
	actFctType(layer.actFctType),
	previousLayer(layer.previousLayer) {

	nodes.reserve(layer.nodes.size());
	for (Layer::Node* node : layer.nodes) {
		nodes.push_back(node);
	}
}

NeuralNetwork::Layer::~Layer() {
	for (Node* node : nodes) {
		delete node;
	}
	nodes.clear();
}

NeuralNetwork::Layer::Node* NeuralNetwork::Layer::getNode(int index) {
	return nodes.at(index);
}

void NeuralNetwork::Layer::calcLayer() {
	for (size_t i = 0; i < nodes.size(); ++i) {
		Node *node = getNode(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

void NeuralNetwork::Layer::calcNodeOutput(Node* node) {

	// Start by adding the bias
	node->output = node->bias;

	for (size_t i = 0; i < previousLayer->nodes.size(); ++i) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}

void NeuralNetwork::Layer::activateNode(Node* node) {
	switch (actFctType) {
	case SIGMOID: {
		node->output = 1.0 / (1.0 + (exp(-node->output)));
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

NeuralNetwork::Layer::Node::Node(const double bias, const double output) :
		Node(0, bias, output) {
}

NeuralNetwork::Layer::Node::Node(const int weightCount, const double bias,
		const double output) :
		bias(bias), output(output) {
	for (int i = 0; i < weightCount; i++)
		weights.push_back(0);
}

NeuralNetwork NeuralNetwork::LoadYAML(string const& path) {

	YAML::Node netDef = YAML::LoadFile(path);

	NeuralNetwork::Layer* inpLayer = nullptr;
	NeuralNetwork::Layer* hidLayer = nullptr;
	NeuralNetwork::Layer* outLayer = nullptr;
	double learningRate = 0.5;

	for (YAML::const_iterator entry = netDef.begin(); entry != netDef.end();
			++entry) {
		string const key = entry->first.as<string>();
		if ("learningRate" == key) {
			learningRate = entry->second.as<double>();
		} else if ("INPUT" == key) {
			inpLayer = Layer::LoadLayer(entry->second, NeuralNetwork::INPUT);
		} else if ("HIDDEN" == key) {
			hidLayer = Layer::LoadLayer(entry->second, NeuralNetwork::HIDDEN);
		} else if ("OUTPUT" == key) {
			outLayer = Layer::LoadLayer(entry->second, NeuralNetwork::OUTPUT);
		}
	}

	hidLayer->previousLayer = inpLayer;
	outLayer->previousLayer = hidLayer;

	NeuralNetwork net;
	net.learningRate = learningRate;
	net.layers.push_back(inpLayer);
	net.layers.push_back(hidLayer);
	net.layers.push_back(outLayer);

	return net;
}

NeuralNetwork::Layer* NeuralNetwork::Layer::LoadLayer(
		YAML::Node const& layerNode, NeuralNetwork::LayerType const layerType) {

	YAML::Node nodeList = layerNode["Nodes"];
	size_t nodeCount = nodeList.size();
	size_t weightCount = 0;
	if (nodeCount > 0) {
		weightCount = nodeList[0]["weights"].size();
	}

	ActFctType actFctType = NONE;
	string const actFctName = layerNode["activationFunction"].as<string>();
	if (actFctName == "SIGMOID") {
		actFctType = NeuralNetwork::SIGMOID;
	} else if (actFctName == "TANH") {
		actFctType = NeuralNetwork::TANH;
	} else if (actFctName == "NONE") {
		actFctType = NeuralNetwork::NONE;
	} else {
		cerr << "Unknown activation function type '" << actFctName
				<< "'. Using NONE." << endl;
	}

	Layer* newLayer = new Layer(layerType, actFctType);
	newLayer->nodes.reserve(nodeCount);
	for (int i = 0; i < nodeList.size(); ++i) {
		NeuralNetwork::Layer::Node* node = new NeuralNetwork::Layer::Node(
				nodeList[i]["bias"].as<double>(), 0.0);
		node->weights.reserve(weightCount);
		for (int j = 0; j < weightCount; ++j) {
			node->weights.push_back(nodeList[i]["weights"][j].as<double>());
		}
		newLayer->nodes.push_back(node);
	}

	return newLayer;
}

bool NeuralNetwork::saveYAML(string const& path) {
	ofstream fout(path);

	fout << *this;

	return true;
}

ostream& operator<<(ostream& out, NeuralNetwork const& net) {

	YAML::Emitter netDef;

	netDef << YAML::BeginMap;
	netDef << YAML::Key << "learningRate";
	netDef << YAML::Value << net.learningRate;
	for (NeuralNetwork::Layer* layer : net.layers) {
		switch (layer->layerType) {
		case NeuralNetwork::INPUT:
			netDef << YAML::Key << "INPUT";
			break;
		case NeuralNetwork::OUTPUT:
			netDef << YAML::Key << "OUTPUT";
			break;
		case NeuralNetwork::HIDDEN:
			netDef << YAML::Key << "HIDDEN";
			break;
		}
		netDef << YAML::Value << YAML::BeginMap;
		// Activation function
		netDef << YAML::Key << "activationFunction";
		switch (layer->actFctType) {
		case NeuralNetwork::SIGMOID:
			netDef << YAML::Value << "SIGMOID";
			break;
		case NeuralNetwork::TANH:
			netDef << YAML::Value << "TANH";
			break;
		case NeuralNetwork::NONE:
			netDef << YAML::Value << "NONE";
			break;
		default:
			cerr << "Activation function type not handled, yet." << endl;
		}
		netDef << YAML::Key << "Nodes";
		netDef << YAML::Value;
		netDef << YAML::BeginSeq;
		for (NeuralNetwork::Layer::Node* node : layer->nodes) {
			// Sequence of nodes
			// Node as a map
			netDef << YAML::BeginMap;
			netDef << YAML::Key << "bias";
			netDef << YAML::Value << node->bias;
			netDef << YAML::Key << "weights";
			netDef << YAML::Flow << node->weights;
			netDef << YAML::EndMap;
		}
		netDef << YAML::EndSeq;
		netDef << YAML::EndMap;
	}
	netDef << YAML::EndMap;

	return out << netDef.c_str();
}
