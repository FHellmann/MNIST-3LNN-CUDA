#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;


enum LayerType {INPUT, HIDDEN, OUTPUT};
enum ActFctType {SIGMOID, TANH};

struct Node {
	double bias;
	double output;
	vector<double> weights;

	/**
	 * Creates a zeroed-out node.
	 *
	 * @param weightCount Number of weights per node.
	 */
	Node(const int weightCount) : bias(0), output(0) {
		this(weightCount, 0, 0);
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

struct Layer {
	const LayerType layerType;
	const ActFctType actFctType;
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
			const ActFctType actFctType
			) : layerType(layerType), actFctType(actFctType) {
		Node* node(weightCount);

		for(int i=0; i < nodeCount; i++)
			nodes.push_back(node);
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
	    Layer *prevLayer = getPrevLayer(layer);

	    // Start by adding the bias
	    node->output = node->bias;

	    for (int i=0; i < prevLayer->nodes.size(); i++){
	        Node *prevLayerNode = getNode(prevLayer, i);
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

class NeuralNetwork {
private:
	double learningRate;
	vector<Layer*> layers;

	/**
	 * Get the layer by type.
	 *
	 * @return the first layer of the network with the specified type of layer
	 * or NULL if none was found.
	 */
	Layer* getLayer(const LayerType layerType) {
		for(int i=0; i < layers.size(); i++) {
			Layer *layer = layers.at(i);
			if(layer->layerType == layerType)
				return layer;
		}
		return NULL;
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
	    Layer *layer = getLayer(OUTPUT);

	    for (int i=0; i < layer->nodes.size(); i++){
	        Node *node = getNode(layer, i);

	        int targetOutput = (i==targetClassification) ? 1 : 0;

	        double errorDelta = targetOutput - node->output;
	        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, node->output);

	        updateNodeWeights(nn, OUTPUT, i, errorSignal);
	    }
	}

	/**
	 * Back propagates error in hidden layer.
	 *
	 * @param targetClassification Correct classification (=label) of the input stream.
	 */
	void backPropagateHiddenLayer(const int targetClassification) {
	    Layer *ol = getLayer(OUTPUT);
	    Layer *layer_hidden = getLayer(HIDDEN);

	    for (int h=0; h < layer_hidden->nodes.size(); h++){
	        Node *hn = getNode(layer_hidden,h);

	        double outputcellerrorsum = 0;

	        for (int o=0; o < ol->nodes.size(); o++){

	            Node *on = getNode(ol,o);

	            int targetOutput = (o==targetClassification)?1:0;

	            double errorDelta = targetOutput - on->output;
	            double errorSignal = errorDelta * ol->getActFctDerivative(on->output);

	            outputcellerrorsum += errorSignal * on->weights[h];
	        }

	        double hiddenErrorSignal = outputcellerrorsum * layer_hidden->getActFctDerivative(hn->output);

	        updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	    }
	}

	/**
	 * Updates a node's weights based on given error.
	 *
	 * @param ltype The nodes of this layer.
	 * @param id Sequential id of the node that is to be calculated.
	 * @param error The error (difference between desired output and actual output).
	 */
	void updateNodeWeights(const LayerType ltype,
			const int id,
			double error) {
		Layer *layer = getLayer(layerType);
		Node *node = getNode(layer, id);
		Layer *prevLayer = getPrevLayer(nn, layer);

		for (int i=0; i < node->weights.size(); i++) {
			Node *prevLayerNode = getNode(prevLayer, i);
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
		layers.push_back(createLayer(inpCount, 0, INPUT, NONE));
		layers.push_back(createLayer(hidCount, inpCount, HIDDEN, SIGMOID));
		layers.push_back(createLayer(outCount, hidCount, OUTPUT, SIGMOID));

		for(int l=0; l < layers.size() - 1; l++) { // leave out the output layer
			Layer* layer = layers.at(l);
			for(int i=0; i < layer->nodes.size(); i++) {
				Node *node = layer->getNode(i);

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
		getLayer(HIDDEN)->calcLayer();
		getLayer(OUTPUT)->calcLayer();
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
	    Layer *layer = getLayer(OUTPUT);

	    double maxOut = 0;
	    int maxInd = 0;

	    for (int i=0; i < layer->nodes.size(); i++){
	        Node *on = getNode(layer, i);

	        if (on->output > maxOut){
	            maxOut = on->output;
	            maxInd = i;
	        }
	    }

	    return maxInd;
	}
};
