#include <string>
#include <ostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "NeuralNetwork.hpp"

using namespace std;

ostream& operator<< (ostream& out, NeuralNetwork const& net) {

	YAML::Emitter netDef;

	netDef << YAML::BeginMap;
	netDef << YAML::Key << "learningRate";
	netDef << YAML::Value << net.learningRate;
	for (Layer* layer : net.layers) {
		switch (layer->layerType) {
			case INPUT:
				netDef << YAML::Key << "INPUT";
				break;
			case OUTPUT:
				netDef << YAML::Key << "OUTPUT";
				break;
			case HIDDEN:
				netDef << YAML::Key << "HIDDEN";
				break;
		}
		netDef << YAML::Value << YAML::BeginMap;
		// Activation function
		netDef << YAML::Key << "activationFunction";
		switch (layer->actFctType) {
		case SIGMOID:
			netDef << YAML::Value << "SIGMOID";
			break;
		case TANH:
			netDef << YAML::Value << "TANH";
			break;
		case NONE:
			netDef << YAML::Value << "NONE";
			break;
		default:
			cerr << "Activation function type not handled, yet." << endl;
		}
		netDef << YAML::Key << "Nodes";
		netDef << YAML::Value;
		netDef << YAML::BeginSeq;
		for (Node* node : layer->nodes) {
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

bool saveNet(string const& path, NeuralNetwork const& net) {
	ofstream fout(path);

	fout << net;

	return true;
}

Layer* loadLayer (YAML::Node const& layerNode, LayerType const layerType) {

	YAML::Node nodeList = layerNode["Nodes"];
	size_t nodeCount = nodeList.size();
	size_t weightCount = 0;
	if (nodeCount > 0) {
		weightCount = nodeList[0]["weights"].size();
	}

	ActFctType actFct = NONE;
	string const actFctName = layerNode["activationFunction"].as<string>();
	if (actFctName == "SIGMOID") {
		actFct = SIGMOID;
	} else if (actFctName == "TANH") {
		actFct = TANH;
	} else if (actFctName == "NONE") {
		actFct = NONE;
	} else {
		cerr << "Unknown activation function type '" << actFctName << "'. Using NONE." << endl;
	}

	Layer* layer = createLayer(nodeCount, weightCount, layerType, actFct);
	for (int i = 0; i < nodeList.size(); ++i) {
		layer->nodes[i]->bias = nodeList[i]["bias"].as<double>();
		for (int j = 0; j < weightCount; ++j) {
			layer->nodes[i]->weights[j] = nodeList[i]["weights"][j].as<double>();
		}
	}

	return layer;
}

NeuralNetwork loadNet(std::string const& path) {

	YAML::Node netDef = YAML::LoadFile(path);

	Layer* inpLayer;
	Layer* hidLayer;
	Layer* outLayer;
	double learningRate = 0.5;

	for (YAML::const_iterator entry = netDef.begin(); entry != netDef.end(); ++entry) {
		string const key = entry->first.as<string>();
		if ("learningRate" == key) {
			learningRate = entry->second.as<double>();
		} else if ("INPUT" == key) {
			inpLayer = loadLayer(entry->second, INPUT);
		} else if ("HIDDEN" == key) {
			hidLayer = loadLayer(entry->second, HIDDEN);
		} else if ("OUTPUT" == key) {
			outLayer = loadLayer(entry->second, OUTPUT);
		}
	}

	return NeuralNetwork(inpLayer, hidLayer, outLayer, learningRate);
}
