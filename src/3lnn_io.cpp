/*
 * 3lnn_io.cpp
 *
 *  Created on: 05.07.2017
 *      Author: Stefan
 */
#include "3lnn_io.h"
#include "3lnn.h"
#include <yaml-cpp/yaml.h>

using namespace std;

bool saveNet(string const& path, Network const& net) {
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
	}
	netDef << YAML::EndMap;

	cout << netDef.c_str() << endl;
	return true;
}
