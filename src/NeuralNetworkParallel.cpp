#include "NeuralNetworkParallel.h"

using namespace std;

NeuralNetworkParallel::NeuralNetworkParallel(const int inpCount,
		const int hidCount, const int outCount, const double _learningRate) {
	learningRate = _learningRate;
	layers.push_back(new LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(
			new LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID,
					layers.back()));
	layers.push_back(
			new LayerParallel(outCount, hidCount, OUTPUT, SIGMOID,
					layers.back()));

	for (int l = 0; l < layers.size() - 1; l++) { // leave out the output layer
		Layer* layer = layers.at(l);
		for (int i = 0; i < layer->nodes.size(); i++) {
			Layer::Node *node = layer->getNode(i);

			for (int j = 0; j < node->weights.size(); j++) {
				node->weights[j] = 0.7 * (rand() / (double) (RAND_MAX));
				if (j % 2)
					node->weights[j] = -node->weights[j]; // make half of the weights negative
			}

			node->bias = rand() / (double) (RAND_MAX);
			if (i % 2)
				node->bias = -node->bias; // make half of the bias weights negative
		}
	}
}

NeuralNetworkParallel::NeuralNetworkParallel(NeuralNetworkParallel const& net) :
		NeuralNetwork(net.learningRate) {
	layers.reserve(net.layers.size());
	for (size_t i = 0; i < net.layers.size(); ++i) {
		// Make a deep copy of the layers
		layers.push_back(
				new LayerParallel(
						*dynamic_cast<LayerParallel*>(net.layers[i])));
	}

	// And set the previous layer in the new network.
	for (size_t i = 0; i < layers.size(); ++i) {
		size_t prevLayerIdx = 0;
		for (; prevLayerIdx < net.layers.size(); ++prevLayerIdx) {
			if (net.layers[prevLayerIdx] == net.layers[i]->previousLayer) {
				break;
			}
		}
		layers[i]->previousLayer = layers[prevLayerIdx];
	}
}

/**
 * Merge the weights of omp_in into the omp_out network. The reset network is used
 * to subtract the weights with the weights of omp_in. This results in only adding
 * the delta weights to the weights of omp_out.
 *
 * @param omp_in The neural network with the weights to add to omp_out.
 * @param omp_out The neural network to add the delta weights to.
 * @param reset The neural network with the weights to subtract from omp_in weights.
 */
void mergeNeuralNetworks(NeuralNetworkParallel& omp_in,
		NeuralNetworkParallel& omp_out,
		NeuralNetworkParallel* reset = nullptr) {
	for (int l = 0; l < omp_in.layers.size(); l++) {
		NeuralNetwork::Layer* layerIn = omp_in.layers.at(l);
		NeuralNetwork::Layer* layerOut = omp_out.layers.at(l);
		NeuralNetwork::Layer* layerReset = (
				reset != nullptr ? reset->layers.at(l) : nullptr);
		for (int n = 0; n < layerIn->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* nodeIn = layerIn->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeOut = layerOut->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeReset = (
					layerReset != nullptr ? layerReset->nodes.at(n) : nullptr);
			for (int w = 0; w < nodeIn->weights.size(); w++) {
				if (nodeReset != nullptr) {
					nodeOut->weights.at(w) += nodeIn->weights.at(w)
							- nodeReset->weights.at(w);
				} else {
					nodeOut->weights.at(w) = nodeIn->weights.at(w);
				}
			}
		}
	}
}

/*
double NeuralNetworkParallel::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {

	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();
	double newError = 0;

	NeuralNetworkParallel nnp_merge(*this);

	// Split the work to n threads
#pragma omp parallel shared(needsFurtherTraining,error,newError)
	{
		// Every thread creates an own copy of the neural network to work on
		NeuralNetworkParallel nnp_local(*this);
		int every_ten_percent = images.size() / 10;

		while (needsFurtherTraining) {
			size_t localErrCount = 0;

#pragma omp for
			for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {
				// Convert the MNIST image to a standardized vector format and feed into the network
				nnp_local.feedInput(images[imgCount]);

				// Feed forward all layers (from input to hidden to output) calculating all nodes' output
				nnp_local.feedForward();

				// Back propagate the error and adjust weights in all layers accordingly
				nnp_local.backPropagate(labels[imgCount]);

				// Classify image by choosing output cell with highest output
				int classification = nnp_local.getNetworkClassification();
				if (classification != labels[imgCount])
					localErrCount++;

				// Display progress during training
				//if ((imgCount % every_ten_percent) == 0) {
				//	log(to_string(imgCount / every_ten_percent * 10.0) + "%", omp_get_thread_num());
				//}
			}

#pragma omp atomic
			newError += static_cast<double>(localErrCount)
					/ static_cast<double>(images.size());

			// Merge network weights together
#pragma omp critical
			mergeNeuralNetworks(nnp_local, nnp_merge, this);

#pragma omp barrier
			// Merge the global weights back to the local networks of each thread
			mergeNeuralNetworks(nnp_merge, nnp_local, &nnp_local);

#pragma omp master
			mergeNeuralNetworks(nnp_merge, *this);

#pragma omp barrier

			if (newError < error) {
				error = newError;
			}

			if (newError < error) {
				error = newError;
			} else if (newError > error + max_derivation) {
				// The error increases again. This is not good.
				needsFurtherTraining = false;
			}

			if (error < training_error_threshold) {
				needsFurtherTraining = false;
			}

#pragma omp master
			{

				log("Error: " + to_string(newError * 100.0) + "%");

				newError = 0;
			}
		}
	}

	return error;
}
*/

void NeuralNetworkParallel::backPropagateOutputLayer(const int targetClassification) {
	Layer *layer = getLayer(OUTPUT);

#pragma omp parallel for shared(layer)
	for (int i = 0; i < layer->nodes.size(); i++) {
		Layer::Node *node = layer->getNode(i);

		int const targetOutput = (i == targetClassification) ? 1 : 0;

		double const errorDelta = targetOutput - node->output;
		double const errorSignal = errorDelta
				* layer->getActFctDerivative(node->output);

		updateNodeWeights(OUTPUT, i, errorSignal);
	}
}

void NeuralNetworkParallel::backPropagateHiddenLayer(const int targetClassification) {
	Layer *ol = getLayer(OUTPUT);
	Layer *layer_hidden = getLayer(HIDDEN);

#pragma omp parallel for shared(ol, layer_hidden)
	for (int h = 0; h < layer_hidden->nodes.size(); h++) {
		Layer::Node *hn = layer_hidden->getNode(h);

		double outputcellerrorsum = 0;

		for (int o = 0; o < ol->nodes.size(); o++) {

			Layer::Node *on = ol->getNode(o);

			int const targetOutput = (o == targetClassification) ? 1 : 0;

			double const errorDelta = targetOutput - on->output;
			double const errorSignal = errorDelta
					* ol->getActFctDerivative(on->output);

			outputcellerrorsum += errorSignal * on->weights[h];
		}

		double const hiddenErrorSignal = outputcellerrorsum
				* layer_hidden->getActFctDerivative(hn->output);

		updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	}
}

void NeuralNetworkParallel::updateNodeWeights(const NeuralNetwork::LayerType layertype,
		const int id, double error) {
	Layer *layer = getLayer(layertype);
	Layer::Node *node = layer->getNode(id);
	Layer *prevLayer = layer->previousLayer;

	for (size_t i = 0; i < node->weights.size(); ++i) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		node->weights.at(i) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
}

void NeuralNetworkParallel::LayerParallel::calcLayer() {
#pragma omp parallel for
	for (size_t i = 0; i < nodes.size(); ++i) {
		Node *node = nodes.at(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

/*
void NeuralNetworkParallel::LayerParallel::calcNodeOutput(Node* node) {

	// Start by adding the bias
	node->output = node->bias;

#pragma omp parallel for
	for (size_t i = 0; i < previousLayer->nodes.size(); ++i) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}
*/

NeuralNetworkParallel::LayerParallel::LayerParallel(const int nodeCount,
		const int weightCount, const LayerType _layerType,
		const ActFctType _actFctType, Layer* _previous) :
		Layer(nodeCount, weightCount, _layerType, _actFctType, _previous) {
}

NeuralNetworkParallel::LayerParallel::LayerParallel(LayerParallel const& layer) :
		Layer(layer.layerType, layer.actFctType) {

	nodes.reserve(layer.nodes.size());
	for (LayerParallel::Node* node : layer.nodes) {
		nodes.push_back(new Node(*node));
	}
}
