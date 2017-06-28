/*
 * 3lnn.h
 *
 *  Created on: 28.06.2017
 *      Author: Hellmann Fabio
 */

#include <vector>

using namespace std;

enum LayerType {INPUT, HIDDEN, OUTPUT};

struct Node {
	double bias;
	double output;
	int wcount;
	vector<double*> weights;
};

struct Layer {
	int ncount;
	vector<Node*> nodes;
};

struct Network {
	int inpNodeSize;
	int inpLayerSize;
	int hidNodeSize;
	int hidLayerSize;
	int outNodeSize;
	int outLayerSize;
	vector<Layer*> layers;
};

Network *createNetwork(int size_of_input_vector,
                       int number_of_nodes_in_hidden_layer,
                       int number_of_nodes_in_output_layer);
