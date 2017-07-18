#include "NeuralNetworkDistributed.h"

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(const int inpCount, const int hidCount,
		const int outCount, const double _learningRate) {
	learningRate = _learningRate;
	layers.push_back(new NeuralNetworkParallel::LayerParallel(inpCount, 0, INPUT, NONE, nullptr));
	layers.push_back(new NeuralNetworkParallel::LayerParallel(hidCount, inpCount, HIDDEN, SIGMOID, layers.back()));
	layers.push_back(new NeuralNetworkParallel::LayerParallel(outCount, hidCount, OUTPUT, SIGMOID, layers.back()));

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

void mergeNeuralNetworks(NeuralNetworkParallel& omp_in, NeuralNetworkParallel& omp_out, NeuralNetworkParallel* reset) {
	for(int l=0; l < omp_in.layers.size(); l++) {
		NeuralNetwork::Layer* layerIn = omp_in.layers.at(l);
		NeuralNetwork::Layer* layerOut = omp_out.layers.at(l);
		NeuralNetwork::Layer* layerReset = reset->layers.at(l);
		for(int n=0; n < layerIn->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* nodeIn = layerIn->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeOut = layerOut->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeReset = layerReset->nodes.at(n);
			for(int w=0; w < nodeIn->weights.size(); w++) {
				nodeOut->weights.at(w) += nodeIn->weights.at(w) - nodeReset->weights.at(w);
			}
		}
	}
}

double* getWeightsByLayer(NeuralNetwork &nn, NeuralNetwork::LayerType type) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	double *weights = new double[layer->nodes[0]->weights.size()];
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			weights[w] = n->weights.at(w);
		}
	}
	return weights;
}

double getGlobalError(int const dataCount, int* errorData, int errorSize) {
	int globalError = 0;
	for(int i=0; i < errorSize; i++) {
		globalError += errorData[i];
	}
	return static_cast<double>(globalError) / static_cast<double>(dataCount);
}

void updateWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* deltaWeights) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			n->weights.at(w) += deltaWeights[w];
		}
	}
	delete[] deltaWeights;
}

void NeuralNetworkDistributed::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
		double const max_derivation) {

	/*
	int argc = 4;
	char** argv = {
			""+getLayer(INPUT)->nodes.size(),
			""+getLayer(HIDDEN)->nodes.size(),
			""+getLayer(OUTPUT)->nodes.size(),
			""+learningRate
	};
	*/

    //MPI_Init(&argc, &argv);
	MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int curr_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if(curr_rank == 0) {
    	// --- MASTER ---

    	bool needsFurtherTraining = true;
    	double error = std::numeric_limits<double>::max();
    	double newError = 0;

    	// 1. Send (MPI_Scatter) training set to slaves
    	uchar **imageArray = new uchar[images.size()][images[0].size];
    	uint8_t *labelArray = new uint8_t[labels.size()];
    	for(int i=0; i < images.size(); i++) {
    		std::copy(images[i].datastart, images[i].dataend, &(imageArray[i][0]));
    		labelArray[i] = labels[i];
    	}
    	MPI_Scatter(&imageArray[0], 1, MPI_UNSIGNED_CHAR, NULL, 0, MPI_INT, curr_rank, MPI_COMM_WORLD);
    	MPI_Scatter(&labelArray[0], 1, MPI_UNSIGNED_CHAR, NULL, 0, MPI_INT, curr_rank, MPI_COMM_WORLD);

    	// 2. Receive (MPI_Gather) initialization information from slaves
    	int *ready = new int[world_size];
    	MPI_Gather(NULL, 0, MPI_INT, &ready[0], 1, MPI_INT, world_rank, MPI_COMM_WORLD);

    	while(needsFurtherTraining) {
			// 3. Send (MPI_Bcast) weights to slaves
			double *weightsHidden = getWeightsByLayer(this, HIDDEN);
			MPI_Bcast(&(getLayer(HIDDEN)->nodes[0]->weights.size()), 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
			MPI_Bcast(&weightsHidden[0], 1, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);

			double *weightsOutput = getWeightsByLayer(this, OUTPUT);
			MPI_Bcast(&(getLayer(OUTPUT)->nodes[0]->weights.size()), 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
			MPI_Bcast(&weightsOutput[0], 1, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);

			// 4. Receive (MPI_Gather) all delta weights from slaves
			double *deltaWeightsHidden = new double[getLayer(HIDDEN)->nodes[0]->weights.size()];
			MPI_Gather(NULL, 0, MPI_INT, &deltaWeightsHidden[0], 1, MPI_INT, world_rank, MPI_COMM_WORLD);

			double *deltaWeightsOutput = new double[getLayer(OUTPUT)->nodes[0]->weights.size()];
			MPI_Gather(NULL, 0, MPI_INT, &deltaWeightsOutput[0], 1, MPI_INT, world_rank, MPI_COMM_WORLD);

			int *errors = new int[world_size];
			MPI_Gather(NULL, 0, MPI_INT, &errors[0], 1, MPI_INT, world_rank, MPI_COMM_WORLD);

			// 5. Check whether stop or repeat (back to 3.)
			updateWeights(this, HIDDEN, deltaWeightsHidden);
			updateWeights(this, OUTPUT, deltaWeightsOutput);

			double const newError = getGlobalError(images.size(), errors, world_size);

			if (newError < error) {
				error = newError;
			}

			if(newError < training_error_threshold || newError > error + max_derivation) {
				needsFurtherTraining = false;
			}

			cout << " Error: " << newError * 100.0 << "%" << endl;
    	}
    } else {
    	// --- SLAVE ---

    	bool needsFurtherTraining = true;

    	// 1. Receive (MPI_Scatter) training set
    	MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, &imageArray, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    	MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, &labelArray, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    	// 2. Send (MPI_Gather) initialization finished
    	MPI_Gather(&(true), 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

    	while(needsFurtherTraining) {
        	double newError = 0;

			// 3. Receive (MPI_Bcast) weights
        	int hiddenWeightsCount = 0;
			MPI_Bcast(&hiddenWeightsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
			double *weightsHidden = new double[hiddenWeightsCount];
			MPI_Bcast(&weightsHidden[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        	int outputWeightsCount = 0;
			MPI_Bcast(&outputWeightsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
			double *weightsOutput = new double[outputWeightsCount];
			MPI_Bcast(&weightsOutput[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// 4. Update Weights
			updateWeights(this, HIDDEN, weightsHidden);
			updateWeights(this, OUTPUT, weightsOutput);

			// 5. Perform training on training set
			NeuralNetworkParallel nnp_merge(*this);

			#pragma omp parallel shared(newError)
			{
				NeuralNetworkParallel nnp_local(*this);
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
				}

				#pragma omp atomic
				newError += static_cast<double>(localErrCount) / static_cast<double>(images.size());

				// merge network weights together
				#pragma omp critical
				mergeNeuralNetworks(nnp_local, nnp_merge, &nnp_merge);
			}

			mergeNeuralNetworks(nnp_merge, *this, this);

			// 6. Send (MPI_Gather) delta weight
			double *deltaWeightsHidden = getWeightsByLayer(this, HIDDEN);
			MPI_Gather(&deltaWeightsHidden[0], 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

			double *deltaWeightsOutput = getWeightsByLayer(this, OUTPUT);
			MPI_Gather(&deltaWeightsOutput[0], 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

			MPI_Gather(&newError, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    	}
    }

	MPI_Finalize();
}
