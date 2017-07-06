#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;

class NeuralNetwork {
public:
	enum ActFctType {
		SIGMOID, TANH, NONE
	};

	enum LayerType {
		INPUT, HIDDEN, OUTPUT
	};

	// Forward declaration of layer class
	class Layer;

	/**
	 * Creates a new simple 3-layer neural network.
	 *
	 * @param inpCount The input layer node count.
	 * @param hidCount the hidden layer node count.
	 * @param outCount The output layer node count.
	 * @param learningRate The learning rate of this neural network.
	 */
	NeuralNetwork(const int inpCount, const int hidCount, const int outCount,
			const double learningRate);

	/**
	 * Creates a simple 3-layer neural network from existing layers.
	 *
	 * @param inpLayer The input layer.
	 * @param hidLayer The hidden layer.
	 * @param outLayer The output layer.
	 * @param learningRate The learning rate of this neural network.
	 */
	NeuralNetwork(Layer* inpLayer, Layer* hidLayer, Layer* outLayer,
			double learningRate);

	/**
	 * Sets v as the input value for the input layers.
	 *
	 * @details v has to match the size of the input layer.
	 */
	/*
	 void feedInput(Vector* v) {
	 Layer *inputLayer = getLayer(INPUT);

	 for (int i=0; i < inputLayer->nodes.size(); i++) {
	 inputLayer->nodes.at(i)->output = v->vals.at(i);
	 }
	 }
	 */

	/**
	 * Feeds input layer values forward to hidden to output layer
	 * (calculation and activation fct).
	 */
	void feedForward();

	/**
	 * Back propagates error from output layer to hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagate(const int targetClassification);

	/**
	 * Get the network's classification using the ID of the node with
	 * the highest output.
	 *
	 * @return the classification of the network.
	 */
	int getNetworkClassification();

	double learningRate;
	vector<Layer*> layers;

	/**
	 * Get the layer by type.
	 *
	 * @return the first layer of the network with the specified type of layer
	 * or NULL if none was found.
	 */
	Layer* getLayer(const LayerType layerType);

	/**
	 * Get the previous layer before the specified one.
	 *
	 * @param thisLayer The Layer which is the next of the searched one.
	 * @return the previous layer.
	 */
	Layer* getPrevLayer(Layer* thisLayer);

	/**
	 * Back propagates error in output layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagateOutputLayer(const int targetClassification);

	/**
	 * Back propagates error in hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagateHiddenLayer(const int targetClassification);

	/**
	 * Updates a node's weights based on given error.
	 *
	 * @param ltype The nodes of this layer.
	 * @param id Sequential id of the node that is to be calculated.
	 * @param error The error (difference between desired output and actual output).
	 */
	void updateNodeWeights(const LayerType layertype, const int id,
			double error);

	class Layer {
	public:

		// Forward declaration of node class.
		class Node;

		const LayerType layerType;
		const ActFctType actFctType;
		Layer* previousLayer;
		vector<Node*> nodes;

		/**
		 * Creates a zeroed-out layer.
		 *
		 * @param nodeCount Number of nodes, obviously.
		 * @param weightCount Number of weights per node.
		 * @param layerType Type of the new layer.
		 * @param actFctType Type of the activation function.
		 */
		Layer(const int nodeCount, const int weightCount,
				const LayerType layerType, const ActFctType actFctType,
				Layer* previous);

		/**
		 * Get a node from the specified index.
		 *
		 * @param index The index of the node.
		 * @return the node.
		 */
		Node* getNode(int index);

		/**
		 * Calculates the new output and (de)activates the nodes.
		 */
		void calcLayer();

		/**
		 * Calculates the new output of a node by passing the values
		 * from the previous nodes through and multiply them with the weights.
		 *
		 * @param node The node which gets the recalulated output.
		 */
		void calcNodeOutput(Node* node);

		/**
		 * Activates the node with a specific algorithm (@see ActFctType).
		 *
		 * @param node The node which will be activated.
		 */
		void activateNode(Node* node);

		/**
		 * Get the derivation of the output value.
		 *
		 * @param outVal The value which was the output of a node.
		 * @return the derivation of the output value.
		 */
		double getActFctDerivative(double outVal);

		class Node {
		public:
			double bias;
			double output;
			vector<double> weights;

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 */
			Node(const int weightCount);

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 * @param bias of this node.
			 * @param output of this node.
			 */
			Node(const int weightCount, const double bias, const double output);
		};
	};
};

#endif
