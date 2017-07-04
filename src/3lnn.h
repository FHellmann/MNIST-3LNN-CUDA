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

/**
 * @details Creates the Neural Network by creating the different layers
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */
Network *createNetwork(const int size_of_input_vector,
		const int number_of_nodes_in_hidden_layer,
		const int number_of_nodes_in_output_layer);

/**
 * Sets v as the input value for the input layers.
 *
 * @details v has to match the size of the input layer.
 */
void feedInput(Network* nn,
		Vector *v);

/**
 * Inits the layer of layerType with random values.
 */
void initWeights(Network* nn,
		const LayerType layerType);

/**
 * Creates a zeroed-out layer.
 *
 * @param nodeCount Number of nodes, obviously.
 * @param weightCount Number of weights per node.
 * @param layerType Type of the new layer.
 * @param actFctType Type of the activation function.
 */
Layer* createLayer(const int nodeCount,
		const int weightCount,
		const LayerType layerType,
		const ActFctType actFctType);

/**
 * @returns Returns the first layer of the network nn of layer type layerType. If no layer with the given type exists, returns NULL.
 */
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

