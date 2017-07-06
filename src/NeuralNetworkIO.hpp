#ifndef NEURAL_NETWORK_IO_HPP_
#define NEURAL_NETWORK_IO_HPP_

#include <string>
#include <ostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "NeuralNetwork.hpp"

using namespace std;

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

bool saveNet(string const& path, NeuralNetwork const& net) {
	ofstream fout(path);

	fout << net;

	return true;
}

NeuralNetwork::Layer* loadLayer(YAML::Node const& layerNode,
		NeuralNetwork::LayerType const layerType) {

	YAML::Node nodeList = layerNode["Nodes"];
	size_t nodeCount = nodeList.size();
	size_t weightCount = 0;
	if (nodeCount > 0) {
		weightCount = nodeList[0]["weights"].size();
	}

	NeuralNetwork::ActFctType actFct = NeuralNetwork::NONE;
	string const actFctName = layerNode["activationFunction"].as<string>();
	if (actFctName == "SIGMOID") {
		actFct = NeuralNetwork::SIGMOID;
	} else if (actFctName == "TANH") {
		actFct = NeuralNetwork::TANH;
	} else if (actFctName == "NONE") {
		actFct = NeuralNetwork::NONE;
	} else {
		cerr << "Unknown activation function type '" << actFctName
				<< "'. Using NONE." << endl;
	}

	NeuralNetwork::Layer* layer = new NeuralNetwork::Layer(nodeCount,
			weightCount, layerType, actFct, nullptr);
	for (int i = 0; i < nodeList.size(); ++i) {
		layer->nodes[i]->bias = nodeList[i]["bias"].as<double>();
		for (int j = 0; j < weightCount; ++j) {
			layer->nodes[i]->weights[j] =
					nodeList[i]["weights"][j].as<double>();
		}
	}

	return layer;
}

NeuralNetwork loadNet(std::string const& path) {

	YAML::Node netDef = YAML::LoadFile(path);

	NeuralNetwork::Layer* inpLayer;
	NeuralNetwork::Layer* hidLayer;
	NeuralNetwork::Layer* outLayer;
	double learningRate = 0.5;

	for (YAML::const_iterator entry = netDef.begin(); entry != netDef.end();
			++entry) {
		string const key = entry->first.as<string>();
		if ("learningRate" == key) {
			learningRate = entry->second.as<double>();
		} else if ("INPUT" == key) {
			inpLayer = loadLayer(entry->second, NeuralNetwork::INPUT);
		} else if ("HIDDEN" == key) {
			hidLayer = loadLayer(entry->second, NeuralNetwork::HIDDEN);
		} else if ("OUTPUT" == key) {
			outLayer = loadLayer(entry->second, NeuralNetwork::OUTPUT);
		}
	}

	hidLayer->previousLayer = inpLayer;
	outLayer->previousLayer = hidLayer;

	return NeuralNetwork(inpLayer, hidLayer, outLayer, learningRate);
}

#endif
