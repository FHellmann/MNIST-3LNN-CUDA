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

	Node* getNode(int index) {
		return nodes.at(index);
	}

	void calcLayer() {
		for (int i=0; i < nodes.size(); i++) {
		    Node *node = getNode(i);
			calcNodeOutput(node);
			activateNode(node);
		}
	}

	void calcNodeOutput(Node* node) {
	    Layer *prevLayer = getPrevLayer(layer);

	    // Start by adding the bias
	    node->output = node->bias;

	    for (int i=0; i < prevLayer->nodes.size(); i++){
	        Node *prevLayerNode = getNode(prevLayer, i);
	        node->output += prevLayerNode->output * node->weights.at(i);
	    }
	}

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
	 * Inits the layer of layerType with random values.
	 */
	void initWeights(const LayerType layerType) {
		Layer *layer = getLayer(layerType);
		if(layer != NULL) {
			for(int i=0; i < layer->nodes.size(); i++) {
				Node *node = layer->getNode(i);

				for (int j=0; j < node->weights.size(); j++){
					node->weights[j] = rand()/(double)(RAND_MAX);
				}

				node->bias = rand()/(double)(RAND_MAX);
			}
		} else {
			cerr << "ERROR: Failed to load Layer of type " << layerType << endl;
		}
	}

	/**
	 * @returns Returns the first layer of the network nn of layer type layerType. If no layer with the given type exists, returns NULL.
	 */
	Layer* getLayer(const LayerType layerType) {
		for(int i=0; i < layers.size(); i++) {
			Layer *layer = layers.at(i);
			if(layer->layerType == layerType)
				return layer;
		}
		return NULL;
	}

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

	void feedForwardNetwork() {
		getLayer(HIDDEN)->calcLayer();
		getLayer(OUTPUT)->calcLayer();
	}

	void backPropagateNetwork(const int targetClassification) {
	    backPropagateOutputLayer(targetClassification);
	    backPropagateHiddenLayer(targetClassification);
	}

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
	            double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);

	            outputcellerrorsum += errorSignal * on->weights[h];
	        }

	        double hiddenErrorSignal = outputcellerrorsum * getActFctDerivative(nn, HIDDEN, hn->output);

	        updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	    }
	}

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

		initWeights(HIDDEN);
		initWeights(OUTPUT);
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
