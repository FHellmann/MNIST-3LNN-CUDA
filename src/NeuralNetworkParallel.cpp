#include "NeuralNetworkParallel.h"

#pragma omp declare reducation(mergeWeights:NeuralNetworkPrallel: \
		for(int l=0; l < omp_in.layers.size(); l++) { \
			Layer* layerIn = omp_in.layers.at(l); \
			Layer* layerOut = omp_out.layers.at(l); \
			for(int n=0; n < layerIn->nodes.size(); n++) { \
				Node* nodeIn = layerIn->nodes.at(n); \
				Node* nodeOut = layerOut->nodes.at(n); \
				for(int w=0; w < nodeIn->weights.size(); w++) { \
					double weightIn = nodeIn->weights.at(w); \
					double weightOutOld = nodeOut->weights.at(w); \
					nodeOut->weights.at(w) += weightIn - weightOutOld; \
				} \
			} \
		} \
)

NeuralNetworkParallel::NeuralNetworkParallel(const int inpCount, const int hidCount,
		const int outCount, const double _learningRate) {

	learningRate = _learningRate;
	layers.push_back(new LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(new LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(new LayerParallel(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

	//#pragma omp parallel for
	for (int l = 0; l < layers.size() - 1; l++) { // leave out the output layer
		Layer* layer = layers.at(l);
		for (int i = 0; i < layer->nodes.size(); i++) {
			Layer::Node *node = layer->getNode(i);

			for (int j = 0; j < node->weights.size(); j++) {
				node->weights[j] = 0.7 * (rand() / (double) (RAND_MAX));
				if(j % 2)
					node->weights[j] = -node->weights[j]; // make half of the weights negative
			}

			node->bias = rand() / (double) (RAND_MAX);
			if(i % 2)
				node->bias = -node->bias; // make half of the bias weights negative
		}
	}
}

void NeuralNetworkParallel::feedInput(cv::Mat const& image) {
	Layer* inputLayer = getLayer(INPUT);
	size_t const numPixels = image.cols * image.rows;

	size_t const loopCount = min(numPixels, inputLayer->nodes.size());
	cv::MatConstIterator_<uint8_t> it = image.begin<uint8_t>();
	//#pragma omp parallel for
	for (int i = 0; i < loopCount; ++i, ++it) {
		inputLayer->nodes[i]->output = ((*it > 128.0) ? 1.0 : 0.0);
	}
}

void NeuralNetworkParallel::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {
	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();
	while (needsFurtherTraining) {

		int every_ten_percent = images.size() / 10;
		size_t errCount = 0;

		NeuralNetworkParallel nnp_copy(this);
		// Loop through all images in the file
		//#pragma omp threadprivate(nnp_copy)
		#pragma omp parallel for //copyin(nnp_copy) reducation(mergeWeights:nnp_copy)
		for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {
			// Convert the MNIST image to a standardized vector format and feed into the network
			nnp_copy.feedInput(images[imgCount]);

			// Feed forward all layers (from input to hidden to output) calculating all nodes' output
			nnp_copy.feedForward();

			// Back propagate the error and adjust weights in all layers accordingly
			nnp_copy.backPropagate(labels[imgCount]);

			// Classify image by choosing output cell with highest output
			int classification = nnp_copy.getNetworkClassification();
			if (classification != labels[imgCount])
				errCount++;

			// Display progress during training
			//displayTrainingProgress(imgCount, errCount, 80);
			//displayImage(&img, lbl, classification, 7,6);
			if ((imgCount % every_ten_percent) == 0)
				cout << "x"; cout.flush();
		}

		double newError = static_cast<double>(errCount) / static_cast<double>(images.size());
		if (newError < error) {
			error = newError;
		} else if (newError > error + max_derivation) {
			// The error increases again. This is not good.
			needsFurtherTraining = false;
		}

		if (error < training_error_threshold) {
			needsFurtherTraining = false;
		}

		cout << " Error: " << error * 100.0 << "%" << endl;
	}

	cout << endl;
}

void NeuralNetworkParallel::backPropagateOutputLayer(const int targetClassification) {
	Layer *layer = getLayer(OUTPUT);
	
	//#pragma omp parallel for
	for (int i = 0; i < layer->nodes.size(); i++) {
		Layer::Node *node = layer->getNode(i);

		int targetOutput = (i == targetClassification) ? 1 : 0;

		double errorDelta = targetOutput - node->output;
		double errorSignal = errorDelta
				* layer->getActFctDerivative(node->output);

		updateNodeWeights(OUTPUT, i, errorSignal);
	}
}

void NeuralNetworkParallel::backPropagateHiddenLayer(const int targetClassification) {
	Layer *ol = getLayer(OUTPUT);
	Layer *layer_hidden = getLayer(HIDDEN);

	//#pragma omp parallel for
	for (int h = 0; h < layer_hidden->nodes.size(); h++) {
		Layer::Node *hn = layer_hidden->getNode(h);

		double outputcellerrorsum = 0;

		//#pragma omp parallel for reduction(+:outputcellerrorsum)
		for (int o = 0; o < ol->nodes.size(); o++) {

			Layer::Node *on = ol->getNode(o);

			int targetOutput = (o == targetClassification) ? 1 : 0;

			double errorDelta = targetOutput - on->output;
			double errorSignal = errorDelta
					* ol->getActFctDerivative(on->output);

			outputcellerrorsum += errorSignal * on->weights[h];
		}

		double hiddenErrorSignal = outputcellerrorsum
				* layer_hidden->getActFctDerivative(hn->output);

		updateNodeWeights(HIDDEN, h, hiddenErrorSignal);
	}
}

void NeuralNetworkParallel::updateNodeWeights(const NeuralNetwork::LayerType layertype,
		const int id, double error) {
	Layer *layer = getLayer(layertype);
	Layer::Node *node = layer->getNode(id);
	Layer *prevLayer = layer->previousLayer;

	//#pragma omp parallel for
	for (int i = 0; i < node->weights.size(); i++) {
		Layer::Node *prevLayerNode = prevLayer->getNode(i);
		node->weights.at(i) += learningRate * prevLayerNode->output * error;
	}

	node->bias += learningRate * error;
}

NeuralNetworkParallel::LayerParallel::LayerParallel(const int nodeCount, const int weightCount,
		const LayerType _layerType, const ActFctType _actFctType, Layer* _previous) :
			Layer(nodeCount, weightCount, _layerType, _actFctType, _previous) {
}

void NeuralNetworkParallel::LayerParallel::calcLayer() {
	//#pragma omp parallel for
	for (int i = 0; i < nodes.size(); i++) {
		Node *node = getNode(i);
		calcNodeOutput(node);
		activateNode(node);
	}
}

void NeuralNetworkParallel::LayerParallel::calcNodeOutput(Node* node) {
	// Start by adding the bias
	node->output = node->bias;

	//#pragma omp parallel for
	for (int i = 0; i < previousLayer->nodes.size(); i++) {
		Node *prevLayerNode = previousLayer->getNode(i);
		node->output += prevLayerNode->output * node->weights.at(i);
	}
}
