#include "NeuralNetworkDistributed.h"

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(const int _worldSize, const int _currRank,
		const int inpCount, const int hidCount,
		const int outCount, const double _learningRate) {
	world_size = _worldSize;
	curr_rank = _currRank;
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

void mergeNeuralNetworks(NeuralNetworkDistributed& omp_in, NeuralNetworkDistributed& omp_out, NeuralNetworkDistributed* reset) {
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

double getGlobalError(int const globalSize, int* errorData, int const errorSize) {
	cout << "Sum global error: ";
	int globalError = 0;
	for(int i=0; i < errorSize; i++) {
		cout << errorData[i] << " + ";
		globalError += errorData[i];
	}
	delete[] errorData;
	cout << " = " << globalError << endl;
	return static_cast<double>(globalError) / static_cast<double>(globalSize);
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

    if(curr_rank == 0) {
    	// --- MASTER ---

    	bool needsFurtherTraining = true;
    	double error = std::numeric_limits<double>::max();

    	// 1. Send (MPI_Scatter) training set and init data to slaves
    	cout << "Master: Send training set and init data to slaves...";

    	int imageCount = images.size();
    	int imageSize = images[0].rows * images[0].cols;
    	uchar **imageArray = new uchar*[imageCount];
    	for(int i=0; i < imageCount; i++)
    		imageArray[i] = new uchar[imageSize];
    	uint8_t *labelArray = new uint8_t[imageCount];
    	for(int i=0; i < images.size(); i++) {
    		std::copy(images[i].datastart, images[i].dataend, &(imageArray[i][0]));
    		labelArray[i] = labels[i];
    	}
		MPI_Bcast(&imageCount, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
    	MPI_Scatter(&imageArray[0], 1, MPI_UNSIGNED_CHAR, NULL, 0, MPI_INT, curr_rank, MPI_COMM_WORLD);
    	for(int i=0; i < imageCount; i++)
    		delete[] imageArray[i];
		delete[] imageArray;
    	MPI_Scatter(&labelArray[0], 1, MPI_UNSIGNED_CHAR, NULL, 0, MPI_INT, curr_rank, MPI_COMM_WORLD);
		delete[] labelArray;

		int weightsHiddenSize = getLayer(OUTPUT)->nodes[0]->weights.size();
		MPI_Bcast(&weightsHiddenSize, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
		int weightsOutputSize = getLayer(OUTPUT)->nodes[0]->weights.size();
		MPI_Bcast(&weightsOutputSize, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);

		cout << "FINISHED!" << endl;

    	// 2. Receive (MPI_Gather) initialization information from slaves
    	cout << "Master: Receive initialization information from slaves...";

    	int *ready = new int[world_size];
    	MPI_Gather(NULL, 0, MPI_INT, &ready[0], 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
		delete[] ready;

		cout << "FINISHED!" << endl;

    	while(needsFurtherTraining) {
			// 3. Send (MPI_Bcast) weights to slaves
        	cout << "Master: Send weights to slaves...";
			double *weightsHidden = getWeightsByLayer(*this, HIDDEN);
			MPI_Bcast(&weightsHidden[0], 1, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
			delete[] weightsHidden;

			double *weightsOutput = getWeightsByLayer(*this, OUTPUT);
			MPI_Bcast(&weightsOutput[0], 1, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
			delete[] weightsOutput;
			cout << "FINISHED!" << endl;

			// 4. Receive (MPI_Gather) all delta weights from slaves
        	cout << "Master: Receive all delta weights from slaves...";
			double *deltaWeightsHidden = new double[(world_size-1)*getLayer(HIDDEN)->nodes[0]->weights.size()];
			MPI_Gather(NULL, 0, MPI_INT, &deltaWeightsHidden[0], 1, MPI_INT, curr_rank, MPI_COMM_WORLD);

			double *deltaWeightsOutput = new double[(world_size-1)*getLayer(OUTPUT)->nodes[0]->weights.size()];
			MPI_Gather(NULL, 0, MPI_INT, &deltaWeightsOutput[0], 1, MPI_INT, curr_rank, MPI_COMM_WORLD);

			int *errors = new int[world_size];
			int my_error = 0;
			MPI_Gather(&my_error, 1, MPI_INT, &errors[0], 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
			cout << "FINISHED!" << endl;

			// 5. Check whether stop or repeat (back to 3.)
        	cout << "Master: Check whether stop or repeat...";
			updateWeights(*this, HIDDEN, deltaWeightsHidden);
			updateWeights(*this, OUTPUT, deltaWeightsOutput);

			double const newError = getGlobalError(images.size(), errors, world_size);

			if (newError < error) {
				error = newError;
			}

			if(newError < training_error_threshold || newError > error + max_derivation) {
				needsFurtherTraining = false;
			}
			cout << "FINISHED!" << endl;

			// 6. Notify slaves to go on or exit
        	cout << "Master: Notify slaves to go on or exit...";
			MPI_Bcast(&needsFurtherTraining, 1, MPI_CXX_BOOL, curr_rank, MPI_COMM_WORLD);
			cout << "FINISHED!" << endl;

			cout << "Master: Error: " << newError * 100.0 << "%" << endl;
    	}
    } else {
    	// --- SLAVE ---

    	bool needsFurtherTraining = true;

    	// 1. Receive (MPI_Scatter) training set and init data
    	cout << "Slave-" << curr_rank << ": Receive training set and init data...";
    	int imageCount;
    	int imageSize;
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    	uchar **imageArray = new uchar*[imageCount];
    	for(int i=0; i < imageCount; i++)
    		imageArray[i] = new uchar[imageSize];
    	uint8_t *labelArray = new uint8_t[imageCount];
    	MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, &imageArray[0], 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    	MPI_Scatter(NULL, 0, MPI_UNSIGNED_CHAR, &labelArray[0], 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    	int matrixSize = imageSize / imageSize;
    	cv::Mat *imageMatrixes = new cv::Mat[imageCount];
    	for(int i=0; i < imageCount; i++)
    		imageMatrixes[i] = cv::Mat(matrixSize, matrixSize, CV_8UC1, imageArray[i]);

    	int hiddenWeightsCount = 0;
		MPI_Bcast(&hiddenWeightsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    	int outputWeightsCount = 0;
		MPI_Bcast(&outputWeightsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

		cout << "FINISHED!" << endl;

    	// 2. Send (MPI_Gather) initialization finished
    	cout << "Slave-" << curr_rank << ": Send initialization finished...";
    	bool finished = true;
    	MPI_Gather(&finished, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    	cout << "FINISHED!" << endl;

    	while(needsFurtherTraining) {
        	double newError = 0;

			// 3. Receive (MPI_Bcast) weights
        	cout << "Slave-" << curr_rank << ": Receive weights...";
			double *weightsHidden = new double[hiddenWeightsCount];
			MPI_Bcast(&weightsHidden[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			double *weightsOutput = new double[outputWeightsCount];
			MPI_Bcast(&weightsOutput[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    	cout << "FINISHED!" << endl;

			// 4. Update Weights
	    	cout << "Slave-" << curr_rank << ": Update Weights...";
			updateWeights(*this, HIDDEN, weightsHidden);
			updateWeights(*this, OUTPUT, weightsOutput);

			delete[] weightsHidden;
			delete[] weightsOutput;
	    	cout << "FINISHED!" << endl;

			// 5. Perform training on training set
	    	cout << "Slave-" << curr_rank << ": Perform training on training set...";
			NeuralNetworkDistributed nnp_merge(*this);

			#pragma omp parallel shared(newError)
			{
				NeuralNetworkDistributed nnp_local(*this);
				size_t localErrCount = 0;

				#pragma omp for
				for (size_t imgCount = 0; imgCount < imageCount; imgCount++) {
					// Convert the MNIST image to a standardized vector format and feed into the network
					nnp_local.feedInput(images[imgCount]);

					// Feed forward all layers (from input to hidden to output) calculating all nodes' output
					nnp_local.feedForward();

					// Back propagate the error and adjust weights in all layers accordingly
					nnp_local.backPropagate(labelArray[imgCount]);

					// Classify image by choosing output cell with highest output
					int classification = nnp_local.getNetworkClassification();
					if (classification != labelArray[imgCount])
						localErrCount++;
				}

				#pragma omp atomic
				newError += static_cast<double>(localErrCount);

				// merge network weights together
				#pragma omp critical
				mergeNeuralNetworks(nnp_local, nnp_merge, &nnp_merge);
			}

			mergeNeuralNetworks(nnp_merge, *this, this);
	    	cout << "FINISHED!" << endl;

			// 6. Send (MPI_Gather) delta weight
	    	cout << "Slave-" << curr_rank << ": Send delta weight...";
			double *deltaWeightsHidden = getWeightsByLayer(*this, HIDDEN);
			MPI_Gather(&deltaWeightsHidden[0], 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
			delete[] deltaWeightsHidden;

			double *deltaWeightsOutput = getWeightsByLayer(*this, OUTPUT);
			MPI_Gather(&deltaWeightsOutput[0], 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
			delete[] deltaWeightsOutput;

			MPI_Gather(&newError, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
	    	cout << "FINISHED!" << endl;

			// 7. Wait for command from master to go on or exit
	    	cout << "Slave-" << curr_rank << ": Wait for command from master to go on or exit...";
			MPI_Bcast(&needsFurtherTraining, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
	    	cout << "FINISHED!" << endl;
    	}
    }
}
