#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include "MNISTDataset.h"
#include "utils/Log.hpp"
#include "utils/MNISTStats.h"

#define NUM_DIGITS 10
#define BATCH_SIZE 1

class NeuralNetwork {
private:
	MNISTStats mnistStats;
public:
	enum ActFctType {
		SIGMOID, TANH, NONE
	};

	enum LayerType {
		INPUT, HIDDEN, OUTPUT
	};

	// Forward declaration of layer class
	class Layer;

	double learningRate;
	std::vector<Layer*> layers;

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
	 * Copy constructor.
	 *
	 * Makes a deep copy of the neural net.
	 */
	NeuralNetwork(NeuralNetwork const&);

	/**
	 * Loads a NeuralNetwork from a given YAML file.
	 */
	virtual void loadYAML(std::string const& path);

	/**
	 * Releases all layers and nodes.
	 */
	virtual ~NeuralNetwork();

	/**
	 * Sets v as the input value for the input layers.
	 *
	 * @details v has to match the size of the input layer.
	 */
	virtual void feedInput(cv::Mat const& image);

	/**
	 * Trains the network with the specified set of images and labels. If the
	 * training error threshold is reached or the error increases over a max
	 * derivation, then the train process will stop.
	 *
	 * @param images The image set to train the network with.
	 * @param labels The label set to train the network with.
	 * @param training_error_threshold The maximum error which should be reached.
	 * @param max_derivation The maximum error after the increases again.
	 */
	virtual double train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);

	/**
	 * Get the network's classification using the ID of the node with
	 * the highest output.
	 *
	 * @return the classification of the network.
	 */
	int getNetworkClassification();

	/**
	 * Saves the network to a YAML file given by path.
	 */
	virtual bool saveYAML(std::string const& path);

	/**
	 * Feeds input layer values forward to hidden to output layer
	 * (calculation and activation fct).
	 */
	virtual void feedForward();

	/**
	 * Back propagates error from output layer to hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	virtual void backPropagate(const int targetClassification);

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
	virtual void backPropagateOutputLayer(const int targetClassification);

	/**
	 * Back propagates error in hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	virtual void backPropagateHiddenLayer(const int targetClassification);

	/**
	 * Updates a node's weights based on given error.
	 *
	 * @param ltype The nodes of this layer.
	 * @param id Sequential id of the node that is to be calculated.
	 * @param error The error (difference between desired output and actual output).
	 */
	virtual void updateNodeWeights(const LayerType layertype, const int id,
			double error);

	class Layer {
	public:

		// Forward declaration of node class.
		class Node;

		const LayerType layerType;
		const ActFctType actFctType;
		Layer* previousLayer;
		std::vector<Node*> nodes;

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
		 * Makes a deep copy of the layer.
		 */
		Layer(Layer const&);

		/**
		 * Releases all the nodes.
		 */
		virtual ~Layer();

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
		virtual void calcLayer();

		/**
		 * Calculates the new output of a node by passing the values
		 * from the previous nodes through and multiply them with the weights.
		 *
		 * @param node The node which gets the recalulated output.
		 */
		virtual void calcNodeOutput(Node* node);

		/**
		 * Activates the node with a specific algorithm (@see ActFctType).
		 *
		 * @param node The node which will be activated.
		 */
		virtual void activateNode(Node* node);

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
			std::vector<double> weights;

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 */
			explicit Node(const int weightCount);

			/**
			 * Creates an empty node.
			 *
			 * @param bias of this node.
			 * @param output of this node.
			 */
			Node(const double bias, const double output);

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 * @param bias of this node.
			 * @param output of this node.
			 */
			Node(const int weightCount, const double bias, const double output);
		};

		/** Used in LoadYAML. */
		static Layer* LoadLayer(YAML::Node const& layerNode,
				LayerType const layerType);
	protected:
		/** Used in LoadLayer. */
		Layer(const LayerType, const ActFctType, Layer* previous = nullptr);
	};

protected:

	/** Used in LoadYAML. */
	NeuralNetwork();

	/** Used in sub class copy constructors. */
	NeuralNetwork(double const learningRate);

	friend std::ostream& operator<<(std::ostream&, NeuralNetwork const&);
};

std::ostream& operator<<(std::ostream& out, NeuralNetwork const& net);

#endif
