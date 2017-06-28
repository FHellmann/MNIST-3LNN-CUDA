#include "3lnn.h"

/**
 * @details Creates the Neural Network by creating the different layers
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */
Network* createNetwork(const int inpCount, const int hidCount, const int outCount){
    // Allocate memory block for the network
    Network *nn = new Network();

	initNetwork(nn, inpCount, hidCount, outCount);

	initWeights(nn, HIDDEN);
	initWeights(nn, OUTPUT);

    return nn;
}

/**
 * @details Initializes the Neural Network by creating the different layers and move them into the Neural Networks memory space
 * @param nn Neural Network which needs to be initialized
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */
void initNetwork(Network* nn, const int inpCount, const int hidCount, const int outCount) {
	nn->layers.push_back(createLayer(inpCount, 0, INPUT, NONE));
	nn->layers.push_back(createLayer(hidCount, inpCount, HIDDEN, SIGMOID));
	nn->layers.push_back(createLayer(outCount, hidCount, OUTPUT, SIGMOID));
}

void initWeights(Network* nn, const LayerType layerType) {
	Layer *layer = getLayer(nn, layerType);
	if(layer != NULL) {
		for(int i=0; i < layer->nodes.size(); i++) {
			Node *node = layer->nodes.at(i);

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
 *
 */
Layer* createLayer(const int nodeCount, const int weightCount, const LayerType layerType, const ActFctType actFctType) {
	Layer* layer = new Layer(layerType, actFctType);

	Node* node = new Node();
	node->bias = 0;
	node->output = 0;
	for(int i=0; i < weightCount; i++)
		node->weights.push_back(0);

	for(int i=0; i < nodeCount; i++)
		layer->nodes.push_back(node);

	return layer;
}

Layer* getLayer(Network* nn, const LayerType layerType) {
	for(int i=0; i < nn->layers.size(); i++) {
		Layer *layer = nn->layers.at(i);
		if(layer->layerType == layerType)
			return layer;
	}
	return NULL;
}

Layer* getPrevLayer(Network* nn, Layer* thisLayer) {
	int position = 0;
	for(int i=0; i < nn->layers.size(); i++) {
		Layer *layer = nn->layers.at(i);
		if(layer == thisLayer) {
			position = i;
			break;
		}
	}
	return nn->layers.at(position - 1);
}

Node* getNode(Layer* layer, int id) {
	return layer->nodes.at(id);
}

void feedInput(Network* nn, Vector* v) {
    Layer *inputLayer = getLayer(nn, INPUT);

    for (int i=0; i < inputLayer->nodes.size(); i++) {
    	inputLayer->nodes.at(i)->output = v->vals.at(i);
    }
}

void feedForwardNetwork(Network* nn) {
	calcLayer(nn, HIDDEN);
	calcLayer(nn, OUTPUT);
}

void calcLayer(Network* nn, const LayerType layerType) {
    Layer *layer = getLayer(nn, layerType);

    for (int i=0; i < layer->nodes.size(); i++){
        calcNodeOutput(nn, layerType, i);
        activateNode(nn, layerType, i);
    }
}

void calcNodeOutput(Network* nn, const LayerType layerType, const int id) {
    Layer *layer = getLayer(nn, layerType);
    Node *node = getNode(layer, id);

    Layer *prevLayer = getPrevLayer(nn, layer);

    // Start by adding the bias
    node->output = node->bias;

    for (int i=0; i < prevLayer->nodes.size(); i++){
        Node *prevLayerNode = getNode(prevLayer, i);
        node->output += prevLayerNode->output * node->weights.at(i);
    }
}

void activateNode(Network *nn, const LayerType ltype, const int id) {
    Layer *layer = getLayer(nn, ltype);
    Node *node = getNode(layer, id);

    ActFctType actFct = layer->actFctType;

    switch (actFct) {
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

void backPropagateNetwork(Network *nn, const int targetClassification) {
    backPropagateOutputLayer(nn, targetClassification);
    backPropagateHiddenLayer(nn, targetClassification);
}

void backPropagateOutputLayer(Network *nn, const int targetClassification) {
    Layer *layer = getLayer(nn, OUTPUT);

    for (int i=0; i < layer->nodes.size(); i++){
        Node *node = getNode(layer, i);

        int targetOutput = (i==targetClassification) ? 1 : 0;

        double errorDelta = targetOutput - node->output;
        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, node->output);

        updateNodeWeights(nn, OUTPUT, i, errorSignal);
    }
}

void backPropagateHiddenLayer(Network *nn, const int targetClassification) {

    Layer *ol = getLayer(nn, OUTPUT);
    Layer *layer_hidden = getLayer(nn, HIDDEN);

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

        updateNodeWeights(nn, HIDDEN, h, hiddenErrorSignal);
    }

}

double getActFctDerivative(Network *nn, const LayerType layerType, double outVal) {
    double dVal = 0;
    ActFctType actFct = getLayer(nn, layerType)->actFctType;

    switch (actFct) {
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

void updateNodeWeights(Network *nn, const LayerType layerType, const int id, double error) {
	Layer *layer = getLayer(nn, layerType);
	Node *node = getNode(layer, id);
	Layer *prevLayer = getPrevLayer(nn, layer);

	for (int i=0; i < node->weights.size(); i++) {
		Node *prevLayerNode = getNode(prevLayer, i);
		node->weights.at(1) += nn->learningRate * prevLayerNode->output * error;
	}

	node->bias += nn->learningRate * error;
}

int getNetworkClassification(Network *nn) {
    Layer *layer = getLayer(nn, OUTPUT);

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

