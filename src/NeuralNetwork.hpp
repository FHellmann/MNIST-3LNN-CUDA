#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;

class NeuralNetwork {
public:
	enum ActFctType {SIGMOID, TANH, NONE};

	struct Layer {

		enum LayerType {INPUT, HIDDEN, OUTPUT};

		struct Node {
			double bias;
			double output;
			vector<double> weights;

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 */
			Node(const int weightCount) : Node(weightCount, 0, 0) {
			}

			/**
			 * Creates a zeroed-out node.
			 *
			 * @param weightCount Number of weights per node.
			 * @param bias of this node.
			 * @param output of this node.
			 */
			Node(const int weightCount,
					const double bias,
					const double output) : bias(bias), output(output) {
				for(int i=0; i < weightCount; i++)
					weights.push_back(0);
			}
		};

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
		Layer(const int nodeCount,
				const int weightCount,
				const LayerType layerType,
				const ActFctType actFctType,
				Layer* previous
				) : layerType(layerType), actFctType(actFctType), previousLayer(previous) {
			for(int i=0; i < nodeCount; i++)
				nodes.push_back(new Node(weightCount));
		}

		/**
		 * Get a node from the specified index.
		 *
		 * @param index The index of the node.
		 * @return the node.
		 */
		Node* getNode(int index) {
			return nodes.at(index);
		}

		/**
		 * Calculates the new output and (de)activates the nodes.
		 */
		void calcLayer() {
			for (int i=0; i < nodes.size(); i++) {
			    Node *node = getNode(i);
				calcNodeOutput(node);
				activateNode(node);
			}
		}

		/**
		 * Calculates the new output of a node by passing the values
		 * from the previous nodes through and multiply them with the weights.
		 *
		 * @param node The node which gets the recalulated output.
		 */
		void calcNodeOutput(Node* node) {

		    // Start by adding the bias
		    node->output = node->bias;

		    for (int i=0; i < previousLayer->nodes.size(); i++){
		        Node *prevLayerNode = previousLayer->getNode(i);
		        node->output += prevLayerNode->output * node->weights.at(i);
		    }
		}

		/**
		 * Activates the node with a specific algorithm (@see ActFctType).
		 *
		 * @param node The node which will be activated.
		 */
		void activateNode(Node* node) {
		    switch (actFctType) {
				case SIGMOID: {
					node->output = 1 / (1 + (exp((double) - node->output)) );
					break;
				}
				case TANH: {
					node->output = tanh(node->output);
					break;
				}
		    }
		}

		/**
		 * Get the derivation of the output value.
		 *
		 * @param outVal The value which was the output of a node.
		 * @return the derivation of the output value.
		 */
		double getActFctDerivative(double outVal) {
		    double dVal = 0;
		    switch (actFctType) {
				case SIGMOID: {
					dVal = outVal * (1-outVal);
					break;
				}
				case TANH: {
					dVal = 1-pow(tanh(outVal),2);
					break;
				}
		    }
		    return dVal;
		}
	};

	double learningRate;
	vector<Layer*> layers;

	/**
	 * Get the layer by type.
	 *
	 * @return the first layer of the network with the specified type of layer
	 * or NULL if none was found.
	 */
	Layer* getLayer(const Layer::LayerType layerType) {
		for(int i=0; i < layers.size(); i++) {
			Layer *layer = layers.at(i);
			if(layer->layerType == layerType)
				return layer;
		}
		return nullptr;
	}

	/**
	 * Get the previous layer before the specified one.
	 *
	 * @param thisLayer The Layer which is the next of the searched one.
	 * @return the previous layer.
	 */
	Layer* getPrevLayer(Layer* thisLayer) {
		int position = 0;
		for(int i=0; i < layers.size(); i++) {
			Layer *layer = layers.at(i);
			if(layer == thisLayer) {
				position = i;
				break;
			}
		}
		return layers.at(position - 1);
	}

	/**
	 * Back propagates error in output layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagateOutputLayer(const int targetClassification) {
	    Layer *layer = getLayer(Layer::OUTPUT);

	    for (int i=0; i < layer->nodes.size(); i++){
	        Layer::Node *node = layer->getNode(i);

	        int targetOutput = (i==targetClassification) ? 1 : 0;

	        double errorDelta = targetOutput - node->output;
	        double errorSignal = errorDelta * layer->getActFctDerivative(node->output);

	        updateNodeWeights(Layer::OUTPUT, i, errorSignal);
	    }
	}

	/**
	 * Back propagates error in hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagateHiddenLayer(const int targetClassification) {
	    Layer *ol = getLayer(Layer::OUTPUT);
	    Layer *layer_hidden = getLayer(Layer::HIDDEN);

	    for (int h=0; h < layer_hidden->nodes.size(); h++){
	        Layer::Node *hn = layer_hidden->getNode(h);

	        double outputcellerrorsum = 0;

	        for (int o=0; o < ol->nodes.size(); o++){

	            Layer::Node *on = ol->getNode(o);

	            int targetOutput = (o==targetClassification)?1:0;

	            double errorDelta = targetOutput - on->output;
	            double errorSignal = errorDelta * ol->getActFctDerivative(on->output);

	            outputcellerrorsum += errorSignal * on->weights[h];
	        }

	        double hiddenErrorSignal = outputcellerrorsum * layer_hidden->getActFctDerivative(hn->output);

	        updateNodeWeights(Layer::HIDDEN, h, hiddenErrorSignal);
	    }
	}

	/**
	 * Updates a node's weights based on given error.
	 *
	 * @param ltype The nodes of this layer.
	 * @param id Sequential id of the node that is to be calculated.
	 * @param error The error (difference between desired output and actual output).
	 */
	void updateNodeWeights(const Layer::LayerType layertype,
			const int id,
			double error) {
		Layer *layer = getLayer(layertype);
		Layer::Node *node = layer->getNode(id);
		Layer *prevLayer = layer->previousLayer;

		for (int i=0; i < node->weights.size(); i++) {
			Layer::Node *prevLayerNode = prevLayer->getNode(i);
			node->weights.at(1) += learningRate * prevLayerNode->output * error;
		}

		node->bias += learningRate * error;
	}

public:
	/**
	 * Creates a new simple 3-layer neural network.
	 *
	 * @param inpCount The input layer node count.
	 * @param hidCount the hidden layer node count.
	 * @param outCount The output layer node count.
	 * @param learningRate The learning rate of this neural network.
	 */
	NeuralNetwork(const int inpCount,
			const int hidCount,
			const int outCount,
			const double learningRate) : learningRate(learningRate) {
		layers.push_back(new Layer(inpCount, 0, Layer::INPUT, NONE, nullptr));
		layers.push_back(new Layer(hidCount, inpCount, Layer::HIDDEN, SIGMOID, layers.back()));
		layers.push_back(new Layer(outCount, hidCount, Layer::OUTPUT, SIGMOID, layers.back()));

		for(int l=0; l < layers.size() - 1; l++) { // leave out the output layer
			Layer* layer = layers.at(l);
			for(int i=0; i < layer->nodes.size(); i++) {
				Layer::Node *node = layer->getNode(i);

				for (int j=0; j < node->weights.size(); j++){
					node->weights[j] = rand()/(double)(RAND_MAX);
				}

				node->bias = rand()/(double)(RAND_MAX);
			}
		}
	}

	/**
	 * Creates a simple 3-layer neural network from existing layers.
	 *
	 * @param inpLayer The input layer.
	 * @param hidLayer The hidden layer.
	 * @param outLayer The output layer.
	 * @param learningRate The learning rate of this neural network.
	 */
	NeuralNetwork(Layer* inpLayer,
			Layer* hidLayer,
			Layer* outLayer,
			double learningRate) : learningRate(learningRate) {
		layers.push_back(inpLayer);
		layers.push_back(hidLayer);
		layers.push_back(outLayer);
	}

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
	void feedForward() {
		getLayer(Layer::HIDDEN)->calcLayer();
		getLayer(Layer::OUTPUT)->calcLayer();
	}

	/**
	 * Back propagates error from output layer to hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagate(const int targetClassification) {
	    backPropagateOutputLayer(targetClassification);
	    backPropagateHiddenLayer(targetClassification);
	}

	/**
	 * Get the network's classification using the ID of the node with
	 * the highest output.
	 *
	 * @return the classification of the network.
	 */
	int getNetworkClassification() {
	    Layer *layer = getLayer(Layer::OUTPUT);

	    double maxOut = 0;
	    int maxInd = 0;

	    for (int i=0; i < layer->nodes.size(); i++){
	        Layer::Node *on = layer->getNode(i);

	        if (on->output > maxOut){
	            maxOut = on->output;
	            maxInd = i;
	        }
	    }

	    return maxInd;
	}
};

#endif
