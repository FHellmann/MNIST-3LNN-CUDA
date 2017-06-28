#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

using namespace std;

enum LayerType {INPUT, HIDDEN, OUTPUT};
enum ActFctType {SIGMOID, TANH, NONE};

struct Node {
	double bias;
	double output;
	vector<double> weights;
};

struct Layer {
	const LayerType layerType;
	const ActFctType actFctType;
	vector<Node*> nodes;

	Layer(const LayerType layerType, const ActFctType actFctType)
	: layerType(layerType), actFctType(actFctType) {
	}
};

struct Network {
	double learningRate;
	vector<Layer*> layers;

	Network() : learningRate(0.5) {
	}
};

// PUBLIC
Network *createNetwork(const int size_of_input_vector,
		const int number_of_nodes_in_hidden_layer,
		const int number_of_nodes_in_output_layer);

void feedInput(Network* nn,
		Vector *v);

// PRIVATE - not needed for public access
void initNetwork(Network* nn,
		const int inpCount,
		const int hidCount,
		const int outCount);

void initWeights(Network* nn,
		const LayerType layerType);

Layer* createLayer(const int nodeCount,
		const int weightCount,
		const LayerType layerType,
		const ActFctType actFctType);

Layer* getLayer(Network* nn,
		const LayerType layerType);

void calcLayer(Network* nn,
		const LayerType layerType);

void calcNodeOutput(Network* nn,
		const LayerType layerType,
		const int id);

void activateNode(Network *nn,
		const LayerType ltype,
		const int id);

void backPropagateNetwork(Network *nn,
		const int targetClassification);

void backPropagateOutputLayer(Network *nn,
		const int targetClassification);

void backPropagateHiddenLayer(Network *nn,
		const int targetClassification);

double getActFctDerivative(Network *nn,
		const LayerType layerType,
		double outVal);

void updateNodeWeights(Network *nn,
		const LayerType ltype,
		const int id,
		double error);

int getNetworkClassification(Network *nn);

